"""
planner.py - B-roll scene planner, v3.

Public API (same signature as v2, so the worker pipeline keeps working):

    planned, warnings, model_used = await plan_scenes(
        segments, total_duration, style, avoid, explicit_scenes=None)

Worker helpers (pure functions, no network calls):

    iter_queries(scene)                                -> query ladder, best first
    rank_candidates(scene, candidates, used_ids=())    -> scored, sorted candidates
    choose_best_candidate(scene, candidates, used_ids=(), min_score=1.0)

What it does
------------
* Timeline: gap-free shots. Lines shorter than MIN_SCENE_DURATION_SEC are merged,
  lines longer than MAX_SHOT_DURATION_SEC are split into several shots so the
  B-roll cuts at a natural pace. Word timings are used when they are available.
* Context-aware disambiguation: polysemous words (cloud, python, bank, ...) are
  resolved from neighbouring lines and from the domain of the whole script.
  A bare keyword hit never decides the meaning.
* Safety: real people and brands are replaced with generic visuals and are never
  queried. Sensitive shots get somber, static cinematography and graphic-content
  negative terms.
* Gemini pipeline (optional): global visual brief, chunked deconstruction and
  chunked cinematography, with retries, timeouts, a time budget, partial-result
  recovery and an in-memory cache. Any failure falls back per shot to the
  rule-based planner.
* Filmable queries. Abstract, business and process words ('future', 'goal',
  'workflow', 'architecture') never become stock searches. A search for 'goal'
  tends to return football goals, and 'future' tends to return space imagery.
  Technology narration is mapped to physical stand-ins instead: a developer at a
  laptop, server racks, a robotic arm.
* Relevance gate. A clip is accepted only if its metadata names the subject of at
  least one query in the shot's ladder. Unrelated clips are rejected even when
  their resolution and duration would otherwise score well. The worker needs no
  change for this.
* A query ladder per shot, plus variety control so nearby shots do not reuse
  the same stock search.
"""

import asyncio
import hashlib
import json
import logging
import math
import random
import re
import threading
import time
import unicodedata
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field
from app.config import settings
from app.models import SceneInput

log = logging.getLogger(__name__)

PLANNER_VERSION = "3.1.0"


# =====================================================================
# 0. CONFIGURATION
# =====================================================================

def _setting(name: str, default: Any) -> Any:
    value = getattr(settings, name, None)
    return default if value is None or value == "" else value


def _planner_config() -> Dict[str, Any]:
    return {
        "min_scene": float(_setting("MIN_SCENE_DURATION_SEC", 2.5)),
        "min_last": float(_setting("MIN_LAST_SCENE_DURATION_SEC", 1.8)),
        "max_shot": float(_setting("MAX_SHOT_DURATION_SEC", 6.0)),
        "model": str(_setting("PLANNER_MODEL", "gemini-2.5-pro")),
        "chunk_size": max(1, int(_setting("PLANNER_CHUNK_SIZE", 20))),
        "concurrency": max(1, int(_setting("PLANNER_MAX_CONCURRENCY", 3))),
        "timeout_ms": int(float(_setting("PLANNER_LLM_TIMEOUT_SEC", 120)) * 1000),
        "retries": max(0, int(_setting("PLANNER_LLM_RETRIES", 4))),
        "budget_sec": float(_setting("PLANNER_LLM_BUDGET_SEC", 900)),
        "cache_size": max(0, int(_setting("PLANNER_CACHE_SIZE", 32))),
    }


def _llm_enabled() -> bool:
    key = str(getattr(settings, "GEMINI_API_KEY", "") or "")
    return (
        str(getattr(settings, "PROVIDER_MODE", "")) == "live"
        and bool(key)
        and "your_" not in key
    )


# =====================================================================
# 1. GENERAL WORD LISTS
# =====================================================================

STOPWORDS: Set[str] = set("""
a about above after again against all also am an and any are aren as at be because been
before being below between both but by can cannot could couldn did didn do does doesn doing
don down during each either else even ever every few for from further had has have having he
her here hers herself him himself his how i if in into is isn it its itself just let me more
most much must my myself no nor not now of off on once only or other ought our ours ourselves
out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up upon us very was we were what when
where whether which while who whom whose why will with within without would wouldn you your
yours yourself yourselves yet one ones two three four five six seven eight nine ten hundred
thousand many
""".split())

# Verbs and filler that carry no visual meaning on their own.
WEAK_WORDS: Set[str] = set("""
make makes made making use uses used using want wants wanted need needs needed going gonna
gotta get gets got getting take takes took taking give gives gave giving look looks looked
looking know knows knew knowing think thinks thought feel feels felt feeling see sees saw
seeing say says said tell tells told talk talks talked really actually basically literally
simply always never sometimes often maybe probably still already things thing stuff lot lots
way ways kind sort bit part parts time times today yesterday tomorrow day days something
anything nothing everything everyone everybody anybody nobody someone
""".split())

# Stock and camera jargon. Never useful as a search term.
CAMERA_NOISE: Set[str] = set("""
shot shots view views angle angles camera cameras footage video videos clip clips scene scenes
cinematic cinematography establishing establish broll hd uhd resolution frame frames style
stock professional
""".split())

# Brand and franchise tokens that must never reach a search query.
BRAND_TOKENS: Set[str] = set("""
tesla nike adidas pepsi coca iphone microsoft google openai chatgpt facebook instagram youtube
netflix musk zuckerberg bezos trump modi hogwarts quidditch voldemort dumbledore potter
""".split())

# Abstract, business and process words a camera cannot film. They never become
# stock keywords. A word that has a physical stand-in is mapped through the concept
# tables instead, so 'automation' still yields a robotic arm.
NON_VISUAL: Set[str] = set("""
future technology technologies goal goals reality idea ideas dream dreams vision mission
seamless seamlessly intelligent clean stunning complex complexity performance high
streamline streamlines streamlined optimize optimized optimization enable enables empower
empowers unlock unlocks bring brings brought deliver delivers delivered create creates creating
achieve achieves improve improves improvement transform transforms transformation drive drives
help helps build builds built closer closest powerful amazing incredible beautiful
automation automated automate automatic architecture architectures system systems
workflow workflows pipeline pipelines processing process processes rendering render rendered
solution solutions service services experience experiences quality approach strategy method
methods feature features benefit benefits result results outcome outcomes impact efficiency
efficient productivity innovation success growth visual visuals side line lines
""".split())

# Broad technology words. In a technology script they still map to a visual, but
# they rank below specific terms, so 'automation' leads the shot rather than 'technology'.
TECH_GENERIC: Set[str] = set("""
technology technologies tech system systems intelligent intelligence processing data
""".split())

# Short words that are still meaningful as search terms.
SHORT_ALLOW: Set[str] = set("""
ai tv vr pc ev xr bus car cat dog sun sea gym bed cup pen art ice fan egg tea map hat bag key
net web app bee
""".split())

# Concrete, filmable nouns. Used to rank keywords when no concept matches.
CONCRETE_NOUNS: Set[str] = set("""
office desk laptop computer phone smartphone screen keyboard monitor tablet camera notebook pen
paper book books coffee tea cup plate food meal breakfast lunch dinner kitchen table chair
window door room bedroom bed house home building buildings street streets road roads car cars
bus train metro station airport plane city town village market shop store factory warehouse farm
field fields garden park forest tree trees flower flowers plant plants river lake sea ocean beach
waves mountain mountains hill valley desert sky clouds cloud sun sunrise sunset moon stars rain
snow wind storm ice fire water light hands hand feet foot face eyes person people man woman child
children family friends dog dogs cat cats bird birds horse animal animals fish money coins coin
cash card chart charts graph data stocks shares clock watch alarm bicycle bike runner gym workout
yoga sports ball team stadium track shoes shoe mirror bathroom shower lamp candle candles library
classroom school student students teacher doctor hospital nurse medicine lab robot robots machine
engine tools tool bridge tower castle temple church ship boat harbor port container containers
cargo truck sand rock rocks stone stones wood wooden glass metal paint art music guitar piano
drum microphone stage crowd audience vegetables vegetable meeting conference
""".split())

# Outdoor and landscape words. Only these take the aerial and wide qualifiers.
OUTDOOR_TOKENS: Set[str] = set("""
city street streets road roads mountain mountains forest river lake sea ocean beach sky clouds
valley desert field fields park garden town village harbor port skyline landscape horizon trees
tree sunrise sunset building buildings station airport stadium track market
""".split())

# Words that make a concept a place (aerial and wide qualifiers work best).
PLACE_TOKENS: Set[str] = set("""
city street streets road roads mountain mountains forest river lake sea ocean beach sky clouds
valley desert field fields park garden building buildings town village station airport harbor
port skyline landscape horizon trees tree sunrise sunset night morning office room kitchen
bedroom library classroom factory warehouse stadium track market
""".split())

# Avoid-list expansion. Keys are singular content words; values are terms to
# drop from queries and to reject candidates that contain.
AVOID_EXPANSIONS: Dict[str, List[str]] = {
    "people": [
        "people", "person", "persons", "man", "men", "woman", "women", "crowd", "crowds",
        "face", "faces", "portrait", "portraits", "selfie", "child", "children", "kid", "kids",
        "family", "couple", "group", "team", "worker", "workers", "player", "players", "athlete",
        "athletes", "girl", "boy", "guy",
    ],
    "face": ["face", "faces", "portrait", "portraits", "selfie", "eyes"],
    "logo": ["logo", "logos", "brand", "brands", "branding", "trademark"],
    "text": ["text", "words", "typography", "subtitles", "caption", "captions", "lettering"],
    "animal": [
        "animal", "animals", "dog", "dogs", "cat", "cats", "bird", "birds", "wildlife", "pet",
        "pets", "horse", "horses",
    ],
    "car": ["car", "cars", "vehicle", "vehicles"],
    "water": ["water", "ocean", "sea", "river", "lake", "rain"],
}

# Domain cues: used to detect the domain of the whole script.
DOMAIN_CUES: Dict[str, Set[str]] = {
    "technology": set("""
        software app apps code coding developer developers computer laptop data server servers
        cloud ai digital online internet tech technology technologies startup device phone screen
        program algorithm database cyber website api robot robots python programming machine learning
        automation automated architecture systems pipeline pipelines workflow workflows engineer
        engineers rendering intelligent
    """.split()),
    "finance": set("""
        money bank banks loan loans stock stocks market markets invest investment investor
        investors profit revenue sales price prices cost economy financial finance budget savings
        income dollar rupee cash wealth business company
    """.split()),
    "nature": set("""
        forest river mountain mountains ocean sea lake tree trees wildlife animal animals bird
        birds jungle rain storm weather sky clouds sun sunrise sunset flowers garden park planet
        earth climate
    """.split()),
    "health": set("""
        health doctor doctors hospital medicine sleep exercise workout fitness diet nutrition
        mental therapy wellness heart body disease
    """.split()),
    "travel": set("""
        travel trip journey city street hotel flight airport tourist tourism beach road train
        station passport explore adventure destination
    """.split()),
    "food": set("""
        food cook cooking recipe restaurant kitchen coffee tea meal breakfast lunch dinner fruit
        vegetable chef bake baking
    """.split()),
    "education": set("""
        student students school college university learn learning lesson teacher class exam study
        studying book books knowledge course
    """.split()),
    "sports": set("""
        game match team player players athlete football cricket basketball training coach stadium
        race runner run running win champion
    """.split()),
    "fantasy": set("""
        magic wizard witch spell wand broomstick castle dragon hogwarts potion enchanted sorcery
        fantasy quidditch
    """.split()),
    "sensitive": set("""
        death tragedy funeral mourning grief victims disaster earthquake flood war crisis tragic
        condolence memorial killed died
    """.split()),
    "lifestyle": set("""
        morning routine habit habits productive productivity success mindset discipline goals
        planning plan wake sleep night mobile phone happy life family friends home relationship
    """.split()),
}

# Broad, atmospheric fallbacks per domain. Always the last rung of the ladder.
DOMAIN_BROAD_POOLS: Dict[str, List[str]] = {
    "technology": ["software developer coding laptop", "server room racks", "code on computer screen"],
    "finance": ["business district", "financial charts", "modern office"],
    "nature": ["nature landscape", "forest sunlight", "sky clouds"],
    "health": ["healthy lifestyle", "calm morning", "wellness nature"],
    "travel": ["city street", "travel landscape", "scenic road"],
    "food": ["kitchen cooking", "food close up", "cafe table"],
    "education": ["library books", "classroom study", "notebook desk"],
    "sports": ["sports stadium", "athlete training", "running track"],
    "fantasy": ["misty forest", "candlelight darkness", "ancient castle"],
    "sensitive": ["candle flame darkness", "rain window glass", "empty park bench"],
    "lifestyle": ["morning sunlight", "calm bedroom", "productive workspace"],
    "business": ["business office", "modern office", "city skyline"],
    "general": ["atmospheric landscape", "city skyline", "nature landscape"],
}


# =====================================================================
# 2. VISUAL DICTIONARIES
# =====================================================================

@dataclass(frozen=True)
class _Sense:
    """One meaning of a polysemous word, with the context cues that select it."""
    cues: frozenset
    visual: str
    sensitive: bool = False


def _S(cues: str, visual: str, sensitive: bool = False) -> _Sense:
    return _Sense(frozenset(cues.split()), visual, sensitive)


# Real people, brands and fictional lore -> generic, filmable replacements.
# The name itself is never queried. Matched on whole phrases, longest first.
NAMED_ENTITY_REPLACEMENTS: List[Tuple[str, str]] = [
    ("elon musk", "tech executive on conference stage"),
    ("musk", "tech executive on conference stage"),
    ("mark zuckerberg", "tech executive on conference stage"),
    ("sam altman", "tech executive on conference stage"),
    ("tim cook", "tech executive on conference stage"),
    ("steve jobs", "tech presenter on keynote stage"),
    ("bill gates", "businessman speaking at conference"),
    ("jeff bezos", "businessman walking in warehouse"),
    ("donald trump", "politician speaking at podium"),
    ("narendra modi", "politician speaking at podium"),
    ("taylor swift", "singer performing on stage"),
    ("harry potter", "ancient gothic stone castle"),
    ("hogwarts", "ancient gothic stone castle"),
    ("quidditch", "drone flying over sports field"),
    ("voldemort", "dark silhouette in misty forest"),
    ("dumbledore", "old wizard reading book by candlelight"),
    ("broomstick", "drone flight over misty forest at dusk"),
    ("wand", "sparkling golden light particles dark background"),
    ("chatgpt", "ai chatbot interface on laptop screen"),
    ("openai", "ai research laboratory computers"),
    ("tesla", "electric vehicle assembly line"),
    ("nike", "unbranded running shoes"),
    ("adidas", "unbranded running shoes"),
    ("coca cola", "glass bottle soda on table"),
    ("pepsi", "soda cans on table"),
    ("iphone", "sleek smartphone on desk"),
    ("microsoft", "office workers at computers"),
    ("google", "person searching on laptop"),
    ("youtube", "video editing timeline on screen"),
    ("netflix", "tv remote on couch"),
    ("instagram", "smartphone social media feed"),
    ("facebook", "smartphone social media feed"),
]
NAMED_SORTED: List[Tuple[str, str]] = sorted(
    NAMED_ENTITY_REPLACEMENTS, key=lambda pair: len(pair[0]), reverse=True
)

# Idioms -> physical metaphors. Phrases are matched after ASCII folding.
IDIOM_VISUALS: List[Tuple[str, str]] = [
    ("time is money", "pocket watch close up on desk"),
    ("sky is the limit", "rocket launch into blue sky"),
    ("picture is worth", "photographer framing shot with camera"),
    ("under the weather", "person resting under blanket indoors"),
    ("break the ice", "two people shaking hands at meeting"),
    ("hit the ground running", "runner starting race on track"),
    ("rome wasn t built", "stone masons laying bricks on wall"),
    ("step by step", "feet walking up stone stairs"),
    ("one step at a time", "feet walking up stone stairs"),
    ("rise and shine", "sunrise over calm lake"),
    ("early bird", "sunrise over empty street"),
    ("every drop counts", "water droplets falling close up"),
    ("end of the tunnel", "bright light at end of tunnel"),
    ("burn the midnight oil", "desk lamp glowing late at night"),
]

# Abstract nouns -> physical metaphors.
ABSTRACT_VISUALS: Dict[str, str] = {
    "trust": "two hands shaking in agreement",
    "patience": "snail crawling on green leaf close up",
    "wisdom": "old books on library shelves",
    "freedom": "birds flying over open field at sunrise",
    "teamwork": "team members collaborating around table",
    "innovation": "engineer prototyping in modern workshop",
    "success": "person celebrating achievement on mountain top",
    "discipline": "athlete training intensely in gym",
    "consistency": "alarm clock ticking on wooden table",
    "motivation": "runner sprinting at sunrise",
    "journey": "hiker walking on mountain trail",
    "challenge": "climber on rock face",
    "hope": "sunlight breaking through clouds over field",
    "courage": "lone person standing at cliff edge",
    "healing": "hands holding small plant sprout",
    "community": "neighbors talking on sunny street",
    "security": "padlock close up on wooden door",
}

# Concrete single words -> filmable visuals. Used when no stronger cue exists.
CONCEPT_VISUALS: Dict[str, str] = {
    "sleep": "person sleeping peacefully in bed",
    "wake": "person waking up in sunlit bedroom",
    "alarm": "alarm clock on bedside table",
    "morning": "sunrise through bedroom window",
    "sunrise": "sunrise over calm horizon",
    "breakfast": "healthy breakfast on kitchen table",
    "coffee": "coffee cup on wooden table",
    "workout": "person exercising in gym",
    "exercise": "person exercising outdoors",
    "yoga": "person doing yoga on mat at sunrise",
    "meditation": "person meditating in quiet room",
    "meditate": "person meditating in quiet room",
    "smartphone": "smartphone on bedside table at night",
    "phone": "smartphone close up in hand",
    "mobile": "smartphone on bedside table at night",
    "laptop": "laptop on desk in home office",
    "computer": "computer screen with code close up",
    "keyboard": "hands typing on keyboard close up",
    "journal": "person writing in journal by window",
    "notebook": "person writing in notebook at desk",
    "reading": "person reading book by window",
    "book": "open book on wooden table",
    "books": "stack of books on library shelf",
    "money": "stack of coins on table close up",
    "savings": "piggy bank on wooden table",
    "budget": "calculator and notebook on desk",
    "city": "busy city street at dusk",
    "traffic": "busy traffic on city road",
    "rain": "rain drops on window glass",
    "forest": "sunlight through forest trees",
    "ocean": "waves crashing on beach",
    "sea": "waves rolling on sea shore",
    "beach": "waves rolling onto sandy beach",
    "mountain": "mountain peaks above clouds",
    "mountains": "mountain peaks above clouds",
    "river": "river flowing through green valley",
    "tree": "tree canopy against blue sky",
    "garden": "garden plants in sunlight",
    "flower": "flower petals close up in sunlight",
    "doctor": "doctor in white coat in clinic",
    "hospital": "hospital corridor with soft light",
    "school": "school classroom with empty desks",
    "classroom": "school classroom with empty desks",
    "teacher": "teacher writing on whiteboard",
    "food": "fresh food on wooden table",
    "vegetables": "fresh vegetables on wooden table",
    "fruit": "fresh fruit on wooden table",
    "kitchen": "kitchen counter with fresh vegetables",
    "car": "car driving on empty road",
    "train": "train moving through countryside",
    "airplane": "airplane flying above clouds",
    "plane": "airplane wing above clouds",
    "ship": "cargo ship at sea",
    "factory": "factory production line machinery",
    "robot": "robotic arm working in factory",
    "robots": "robotic arm working in factory",
    "chart": "analyst reviewing chart on laptop",
    "charts": "analyst reviewing charts on laptop",
    "office": "modern office workspace with laptop",
    "meeting": "colleagues talking around conference table",
    "startup": "young team working in bright office",
    "market": "stock market chart on screen",
    "stocks": "stock market chart on screen",
    "music": "hands playing piano keys close up",
    "guitar": "hands playing guitar close up",
    "camera": "camera lens close up",
    "sports": "athlete running on track at sunrise",
    "running": "runner on trail at sunrise",
    "runner": "runner on trail at sunrise",
    "cycling": "cyclist riding on country road",
    "swimming": "swimmer in pool close up",
    "shoes": "running shoes on track close up",
    "water": "water being poured into glass close up",
    "bottle": "water bottle on gym floor",
    "shower": "water running from shower head",
    "mirror": "reflection in bathroom mirror at morning",
    "candle": "candle flame in darkness",
    "candles": "candle flame in darkness",
    "library": "old library bookshelves in soft light",
    "winter": "snow covered trees in winter",
    "snow": "snowflakes falling close up",
    "night": "city lights at night",
    "stars": "starry night sky over mountains",
    "moon": "full moon over quiet lake",
    "space": "earth from space at night",
    "rocket": "rocket launch into blue sky",
    "coins": "stack of coins on table close up",
    "gold": "gold bars close up",
    "house": "modern house exterior at dusk",
    "home": "cozy home interior at dusk",
    "bedroom": "cozy bedroom with soft morning light",
    "window": "sunlight through window curtains",
    "road": "empty road stretching ahead",
    "bridge": "bridge over river at sunrise",
    "building": "modern glass office building",
    "buildings": "modern city skyline at dusk",
    "bicycle": "bicycle parked on city street",
    "bike": "bicycle parked on city street",
    "dog": "dog running in grass",
    "dogs": "dogs running in grass",
    "cat": "cat resting on windowsill",
    "cats": "cats resting on windowsill",
    "bird": "bird flying over lake",
    "birds": "birds flying over lake",
    "fish": "fish swimming in clear water",
    "horse": "horse running across open field",
    "wildlife": "wild animal walking through grassland",
    "hands": "hands close up in natural light",
    "hand": "hand close up in natural light",
    # Technology and software terms. These mean the same thing in any script.
    "developer": "software developer coding laptop",
    "programmer": "programmer coding on laptop",
    "engineer": "engineers collaborating modern office",
    "code": "code on computer screen",
    "coding": "code on computer screen",
    "programming": "code on computer screen",
    "algorithm": "code on computer screen",
    "software": "software interface on laptop screen",
    "app": "smartphone app interface on screen",
    "server": "server room racks blinking lights",
    "database": "server room racks blinking lights",
    "automation": "robotic arm working in factory",
    "automated": "robotic arm working in factory",
    "automatic": "robotic arm working in factory",
    "workflow": "sticky notes planning board",
    "dashboard": "analytics dashboard on monitor",
    "ai": "robot hand digital interface",
    "cyber": "digital padlock on screen",
}

# Technology terms that apply only when the whole script is about technology.
# 'system', 'data' or 'architecture' mean other things in other scripts.
TECH_CONTEXT_VISUALS: Dict[str, str] = {
    "technology": "circuit board closeup",
    "technologies": "circuit board closeup",
    "tech": "circuit board closeup",
    "circuit": "circuit board closeup",
    "architecture": "engineers drawing diagram whiteboard",
    "system": "control room monitors",
    "intelligent": "robot hand digital interface",
    "intelligence": "robot hand digital interface",
    "processing": "computer processor chip closeup",
    "chip": "computer processor chip closeup",
    "rendering": "editing software timeline",
    "render": "editing software timeline",
    "pipeline": "fiber optic cables glowing",
    "network": "network cables closeup",
    "data": "data charts computer monitor",
}

# Polysemous words. Each sense lists the context cues that select it. The
# default sense is used only when no cue appears in the neighbouring lines or
# in the script as a whole. Sensitive senses are never chosen by accident.
POLYSEMY_SENSES: Dict[str, Tuple[str, Dict[str, _Sense]]] = {
    "cloud": ("computing", {
        "computing": _S(
            "server servers data storage saas aws azure backup software digital computing api "
            "hosting datacenter database sync upload download deploy app apps device devices "
            "subscription tech technology",
            "datacenter server racks blinking lights"),
        "weather": _S(
            "sky rain rainy storm weather sunny overcast thunder forecast sunset sunrise blue "
            "dark sun fog mist",
            "dramatic clouds drifting across blue sky"),
    }),
    "clouds": ("weather", {
        "weather": _S(
            "sky rain rainy storm weather sunny overcast thunder forecast sunset sunrise blue "
            "dark sun fog mist",
            "dramatic clouds drifting across blue sky"),
        "computing": _S(
            "server servers data storage saas aws azure backup software digital computing api "
            "hosting datacenter",
            "datacenter server racks blinking lights"),
    }),
    "python": ("programming", {
        "programming": _S(
            "code coding programming programmer developer developers software script scripts "
            "library libraries function functions variable syntax django flask pip ide laptop "
            "computer bug debugging language tech data machine learning ai",
            "programmer coding software on laptop screen"),
        "animal": _S(
            "snake snakes reptile reptiles jungle rainforest venom constrictor zoo species scales "
            "wildlife forest",
            "python snake coiled on tree branch in rainforest"),
    }),
    "mouse": ("computing", {
        "computing": _S(
            "click clicks cursor keyboard desktop computer pc laptop usb wireless scroll pointer "
            "screen monitor workstation office",
            "ergonomic computer mouse on desk"),
        "animal": _S(
            "cheese field rodent rodents trap traps cat cats wildlife whiskers burrow pest pests "
            "lab tail forest grain nest",
            "small field mouse in grass close up"),
    }),
    "bank": ("finance", {
        "finance": _S(
            "money loan loans account accounts deposit deposits interest savings credit finance "
            "financial investment invest branch atm mortgage lender cash currency dollar rupee "
            "payments transaction customers economy business",
            "modern glass bank building facade"),
        "river": _S(
            "river rivers water shore shores fishing flood stream lake erosion sand riverside "
            "creek canal grass mud muddy waterside",
            "riverbank grass beside flowing water"),
    }),
    "platform": ("technology", {
        "technology": _S(
            "app apps software developer developers users saas cloud build launch social online "
            "product products ecosystem api creators digital website startup tech mobile phone "
            "marketplace",
            "smartphone app interface on screen"),
        "train": _S(
            "train trains station railway rail commuter commuters track tracks passengers "
            "departure arrival depot timetable",
            "empty train platform at dawn"),
    }),
    "spell": ("letters", {
        "letters": _S(
            "letter letters word words write writing spelling spelled typed typing correct "
            "dictionary check grammar text keyboard",
            "typing letters on keyboard close up"),
        "magic": _S(
            "magic wizard witch potion wand enchanted ancient curse hogwarts sorcerer fantasy "
            "charm incantation grimoire candle castle",
            "glowing candlelight ancient spellbook"),
    }),
    "value": ("business", {
        "business": _S(
            "revenue price prices profit market investors investor quality growth worth roi cost "
            "costs brand sales customers company business financial money return returns pricing "
            "product products",
            "business analyst reviewing value chart"),
    }),
    "crane": ("construction", {
        "construction": _S(
            "construction site sites building buildings tower towers steel concrete lift lifting "
            "project skyscraper contractor cargo port harbor city",
            "construction crane against blue sky"),
        "bird": _S(
            "bird birds wings wetland wetlands migration migrate lake marsh heron flock species "
            "feathers wildlife nature pond river",
            "crane bird wading in wetland"),
    }),
    "bat": ("animal", {
        "animal": _S(
            "cave caves night nocturnal wings wildlife colony mammal mammals fly flying roost "
            "roosting forest insect insects",
            "bats hanging in dark cave"),
        "sports": _S(
            "baseball cricket pitcher pitch pitches ball league innings stadium game games swing "
            "batter player players match field hit team teams",
            "batter swinging bat on sports field"),
    }),
    "apple": ("fruit", {
        "fruit": _S(
            "fruit fruits orchard orchards tree trees pie juice harvest eat eating ate red green "
            "farm farms picked picking sweet sliced basket market cider",
            "red apples on wooden table"),
        "brand": _S(
            "iphone mac macbook ipad ios device devices store stores watch keynote product "
            "products launch phone phones laptop laptops company tech technology ceo software",
            "sleek modern laptop on desk"),
    }),
    "crash": ("market", {
        "market": _S(
            "market markets stocks stock shares investors economy trading traders bubble index "
            "crypto bitcoin bonds nifty sensex crisis economic financial money",
            "stock market chart falling on screen"),
        "software": _S(
            "server servers app apps software system systems bug bugs update updates reboot laptop "
            "computer website outage code data deploy",
            "laptop screen error close up"),
        "vehicle": _S(
            "car cars road roads highway driver drivers accident collision vehicle vehicles traffic "
            "truck bus speed brake brakes police ambulance injured",
            "empty road at dusk", True),
    }),
    "war": ("sensitive", {
        "sensitive": _S(
            "conflict soldiers soldier troops battle battles attack attacks army military refugees "
            "bombing bombs peace ceasefire violence",
            "candle flame in darkness", True),
        "trade": _S(
            "trade tariff tariffs export exports import imports economy economic market markets "
            "china supply shipping ports cargo sanctions dollar currency",
            "shipping containers stacked at port"),
    }),
    "stream": ("water", {
        "water": _S(
            "river rivers water flows flowing mountain forest creek waterfall fish fishing nature "
            "rocks valley brook lake rain",
            "clear mountain stream flowing over rocks"),
        "creator": _S(
            "live streaming streamer streamers video videos audience viewers platform channel "
            "channels creator creators twitch youtube subscribers broadcast content online followers "
            "gaming gamers podcast",
            "content creator filming live on camera"),
    }),
    "bear": ("animal", {
        "animal": _S(
            "forest wildlife wild fur claws cub cubs woods mountain mountains honey salmon hibernate "
            "hibernation grizzly park trail hunting animal animals nature",
            "brown bear walking through forest"),
        "market": _S(
            "market markets stocks stock shares investors trading traders economy bearish index "
            "prices sell selloff recession downturn financial nifty sensex falling decline",
            "falling stock market chart on screen"),
    }),
    "bull": ("animal", {
        "animal": _S(
            "farm farmland field fields horns ranch cattle cow cows pasture rodeo animal animals",
            "bull standing in open field"),
        "market": _S(
            "market markets stocks stock shares investors trading traders economy bullish index "
            "prices rally rising growth financial nifty sensex",
            "rising stock market chart on screen"),
    }),
}

# Sensitive content: tragedy, death, disaster, grief, violence.
SENSITIVE_TERMS: Set[str] = set("""
death died dies dead funeral funerals mourning grief grieving tragedy tragic victim victims
condolence condolences memorial obituary killed killing massacre bereaved suicide disaster
disasters earthquake tsunami casualties casualty injured injuries explosion shooting hostage
terror terrorist terrorism cancer sad sadness darkness crisis accident
""".split())

# Respectful, non-graphic visuals for sensitive shots. Chosen by stable hash.
SENSITIVE_POOL: List[str] = [
    "candle flame in darkness",
    "rain drops on window glass",
    "empty park bench in autumn",
    "still lake at dawn mist",
    "silhouette on hilltop at dusk",
    "wilted flowers on wooden table",
    "old chapel window soft light",
    "footsteps on quiet empty street",
]

# Graphic content that must never appear in a sensitive shot.
SENSITIVE_NEGATIVES: List[str] = [
    "blood", "bloody", "wound", "wounded", "injury", "injured", "corpse", "corpses", "gore",
    "wreckage", "explosion", "explosions", "weapon", "weapons", "gun", "guns", "bodies",
    "casualty", "casualties",
]

# Imagery that a stock search pulls in from ordinary words ('future' -> planets,
# 'goal' -> football). Rejected unless the script itself is about that subject.
SPACE_WORDS: Set[str] = set("""
space planet planets galaxy galaxies astronomy astronaut astronauts cosmos cosmic universe
rocket rockets orbit mars telescope alien aliens
""".split())
SPACE_NEGATIVES: List[str] = [
    "space", "planet", "planets", "galaxy", "galaxies", "nebula", "cosmic", "universe",
    "alien", "aliens", "astronaut", "astronauts",
]
SPORT_WORDS: Set[str] = set("""
soccer football basketball cricket tennis athlete athletes stadium sports sport coach
goalkeeper championship league marathon olympic olympics
""".split())
SPORT_NEGATIVES: List[str] = [
    "soccer", "football", "goalkeeper", "goalkeepers", "basketball", "cricket", "tennis", "stadium",
]

# Statistics: a numeric pattern (percent, currency, multiplier, magnitude).
STAT_RE = re.compile(
    r"[$₹€£]\s?\d|\b\d[\d,.]*\s?(?:%|percent|per cent|x\b|times\b|million|billion|trillion|bn\b"
    r"|crore|crores|lakh|lakhs|k\b|m\b)",
    re.IGNORECASE,
)
STAT_UP: Set[str] = set("""
grew grow grows growing growth rose rise rises rising increase increased increases surge surged
surges soar soared climb climbed jump jumped boost boosted gain gained doubled tripled record
""".split())
STAT_DOWN: Set[str] = set("""
fell fall falls falling drop dropped drops decline declined declines decrease decreased plunge
plunged shrink shrank slump slumped slid lost loss losses
""".split())
STAT_VISUALS: Dict[str, str] = {
    "up": "analyst pointing at rising bar chart on monitor",
    "down": "declining line chart on digital screen",
    "flat": "business analyst reviewing financial charts on laptop",
}
TREND_LABEL: Dict[str, str] = {"up": "rising", "down": "declining", "flat": "neutral"}

# Tamil and Hindi narration -> English gloss, for offline use. Live mode asks
# Gemini to translate instead. Keys are matched exactly, or by stem (the key
# minus its last character) for inflected forms.
TAMIL_LEXICON: Dict[str, str] = {
    "காலையில்": "morning sunrise alarm clock",
    "எழுவது": "waking up alarm clock",
    "மாற்றங்கள்": "person stretching sunlight",
    "நாளை": "person stretching sunlight",
    "இரவில்": "smartphone bedside night",
    "மொபைல்": "smartphone bedside night",
    "தூக்கமும்": "person sleeping bed",
    "திட்டமிடலும்": "planning notebook desk",
    "பணம்": "money coins",
    "வீடு": "home house",
    "தண்ணீர்": "water glass",
    "மழை": "rain window",
    "சூரியன்": "sun sunrise",
    "கடல்": "sea waves",
    "மக்கள்": "people crowd",
    "பள்ளி": "school classroom",
    "மருத்துவர்": "doctor clinic",
    "உணவு": "healthy food",
    "பயணம்": "travel journey road",
    "நகரம்": "city street",
    "மரம்": "tree forest",
    "குழந்தை": "child playing",
    "தொழில்நுட்பம்": "software developer coding laptop",
    "கணினி": "computer laptop",
    "வேலை": "office work desk",
    "வணிகம்": "business office",
    "பங்குச் சந்தை": "stock market chart",
    "ஆரோக்கியம்": "healthy lifestyle",
    "உடற்பயிற்சி": "exercise workout gym",
    "புத்தகம்": "open book",
    "படிப்பு": "study books library",
}
HINDI_LEXICON: Dict[str, str] = {
    "सुबह": "morning sunrise alarm clock",
    "बदलाव": "person stretching sunlight",
    "रात": "smartphone bedside night",
    "मोबाइल": "smartphone bedside night",
    "नींद": "person sleeping bed",
    "सफलता": "person celebrating achievement",
    "पैसा": "money coins",
    "घर": "home house",
    "पानी": "water glass",
    "बारिश": "rain window",
    "सूरज": "sun sunrise",
    "समुद्र": "sea waves",
    "लोग": "people crowd",
    "स्कूल": "school classroom",
    "डॉक्टर": "doctor clinic",
    "खाना": "healthy food",
    "यात्रा": "travel journey road",
    "शहर": "city street",
    "पेड़": "tree forest",
    "बच्चा": "child playing",
    "तकनीक": "software developer coding laptop",
    "कंप्यूटर": "computer laptop",
    "काम": "office work desk",
    "व्यापार": "business office",
    "शेयर बाजार": "stock market chart",
    "स्वास्थ्य": "healthy lifestyle",
    "व्यायाम": "exercise workout gym",
    "किताब": "open book",
    "पढ़ाई": "study books library",
}

# Used when a non-English line yields no concept at all.
MULTILINGUAL_DEFAULTS: List[str] = [
    "morning sunrise alarm clock",
    "person stretching sunlight",
    "modern workspace desk lamp",
    "productive office laptop coffee",
    "aerial city skyline timelapse",
]

# Shot sizes used to rotate framing across neighbouring shots.
SHOT_ROTATION: List[str] = [
    "Medium Shot",
    "Close-Up",
    "Wide Establishing Shot",
    "Insert Detail",
    "Low-Angle Medium Shot",
    "Overhead Top-Down Shot",
    "Tracking Side Shot",
    "Extreme Wide Aerial Shot",
]

# Close synonyms, used by the subject check so that a clip which says 'programmer'
# or 'robot' still counts as a match for 'developer' or 'robotic'. Keys are stems.
TERM_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "developer": ("programmer", "coder", "coding", "code"),
    "programmer": ("developer", "coder", "coding", "code"),
    "code": ("coding", "programming", "programmer", "developer"),
    "coding": ("code", "programming", "programmer"),
    "programming": ("code", "coding"),
    "software": ("code", "coding", "programming", "app"),
    "server": ("datacenter", "data center", "rack"),
    "rack": ("server", "datacenter"),
    "robot": ("robotic", "automation", "automated"),
    "robotic": ("robot", "automation", "automated"),
    "automation": ("robot", "robotic", "automated"),
    "automated": ("automation", "robot", "robotic"),
    "screen": ("monitor", "display", "laptop"),
    "monitor": ("screen", "display"),
    "laptop": ("computer", "notebook", "screen"),
    "computer": ("laptop", "pc", "monitor"),
    "whiteboard": ("blackboard", "board", "diagram"),
    "board": ("whiteboard", "blackboard"),
    "circuit": ("chip", "electronics"),
    "chip": ("processor", "circuit"),
    "processor": ("chip", "circuit"),
    "cable": ("fiber", "wire", "network"),
    "fiber": ("cable", "optic"),
    "network": ("cable", "server"),
    "dashboard": ("chart", "analytics"),
    "chart": ("dashboard", "graph"),
    "timeline": ("editing", "editor"),
    "editing": ("editor", "timeline"),
    "ai": ("robot", "artificial", "intelligence"),
    "intelligence": ("ai", "artificial"),
    "interface": ("screen", "digital"),
    "smartphone": ("phone", "app"),
    "app": ("smartphone", "phone", "interface"),
    "engineer": ("developer", "programmer"),
    "data": ("chart", "dashboard"),
}

# Adjectives and framings that say nothing about the subject. They never count
# as a subject match on their own.
GENERIC_QUERY_WORDS: Set[str] = set("""
close closeup detail macro wide aerial timelapse modern calm atmospheric cinematic natural
bright soft beautiful stunning empty busy scenic view digital tech
""".split())


def _has_subject(query: str) -> bool:
    """True if the query has at least one word that names something filmable."""
    return any(token not in GENERIC_QUERY_WORDS for token in query.split())


# Qualifiers used to build variant queries for the same concept.
QUAL_PLACE: List[str] = ["aerial", "wide", "timelapse"]
QUAL_OBJECT: List[str] = ["closeup", "detail", "macro"]

# Backward-compatible exports from v2.
NAMED_ENTITIES_OR_BRANDS: List[str] = [phrase for phrase, _ in NAMED_ENTITY_REPLACEMENTS]
SENSITIVE_KEYWORDS: List[str] = sorted(SENSITIVE_TERMS)
DOUBLE_MEANING_HINTS: Dict[str, str] = {
    word: senses[1][senses[0]].visual for word, senses in POLYSEMY_SENSES.items()
}


# =====================================================================
# 3. TEXT UTILITIES
# =====================================================================

_COMBINING = re.compile(r"[\u0300-\u036f]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LETTERS = re.compile(r"[a-z]+")
_SCRIPT_RUN = re.compile(r"[\u0900-\u097F\u0B80-\u0BFF]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _fold(text: str) -> str:
    """Lowercase ASCII text: accents removed, every other character becomes a space."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = _COMBINING.sub("", text).lower()
    return _NON_ALNUM.sub(" ", text).strip()


def _words(text: str) -> List[str]:
    return _LETTERS.findall(_fold(text))


def _detect_language(text: str) -> str:
    """Return 'en' (any Latin script), 'ta' (Tamil), 'hi' (Devanagari) or 'other'."""
    counts: Counter = Counter()
    for ch in str(text or ""):
        if not ch.isalpha():
            continue
        code = ord(ch)
        if 0x0B80 <= code <= 0x0BFF:
            counts["ta"] += 1
        elif 0x0900 <= code <= 0x097F:
            counts["hi"] += 1
        elif ch.isascii() or unicodedata.name(ch, "").startswith("LATIN"):
            counts["en"] += 1
        else:
            counts["other"] += 1
    return counts.most_common(1)[0][0] if counts else "en"


def _lexicon_lookup(token: str, lexicon: Dict[str, str]) -> Optional[str]:
    if token in lexicon:
        return lexicon[token]
    for key, gloss in lexicon.items():
        if " " not in key and len(key) >= 3 and token.startswith(key[:-1]):
            return gloss
    return None


def _lexicon_gloss(text: str) -> Tuple[str, int]:
    """Offline English gloss for Tamil or Hindi text. Returns (gloss, number of hits)."""
    working = str(text or "")
    lexicons = (TAMIL_LEXICON, HINDI_LEXICON)
    phrases: List[str] = []
    for lexicon in lexicons:
        for key in sorted(lexicon, key=len, reverse=True):
            if " " in key and key in working:
                phrases.append(lexicon[key])
                working = working.replace(key, " ")
    for token in _SCRIPT_RUN.findall(working):
        for lexicon in lexicons:
            gloss = _lexicon_lookup(token, lexicon)
            if gloss:
                phrases.append(gloss)
                break
    words: List[str] = []
    for phrase in phrases:
        for word in phrase.split():
            if word not in words:
                words.append(word)
    return " ".join(words), len(phrases)


def _proper_nouns(text: str) -> Set[str]:
    """Capitalised words that are not sentence-initial, i.e. likely names.

    Returns an empty set when the text is mostly Title Case, because then the
    capitalisation carries no information.
    """
    names: List[str] = []
    total = 0
    for sentence in _SENTENCE_SPLIT.split(str(text or "").strip()):
        parts = re.findall(r"[A-Za-z][A-Za-z'\-]*", sentence)
        for index, word in enumerate(parts):
            if index == 0:
                continue
            total += 1
            if word[0].isupper() and not word.isupper():
                names.append(word.lower())
    if total == 0 or len(names) / total > 0.6:
        return set()
    return set(names)


def _is_content_token(token: str) -> bool:
    if token in STOPWORDS or token in WEAK_WORDS or token in CAMERA_NOISE or token in BRAND_TOKENS:
        return False
    if token in NON_VISUAL:
        return False
    return len(token) >= 3 or token in SHORT_ALLOW


def _sanitize_tokens(raw: str, drop: Iterable[str] = (), max_tokens: int = 4) -> str:
    """Turn free text into a clean stock query of at most max_tokens English words."""
    drop_set = drop if isinstance(drop, (set, frozenset)) else set(drop)
    out: List[str] = []
    for token in _LETTERS.findall(_fold(raw)):
        if not _is_content_token(token) or token in drop_set or token in out:
            continue
        out.append(token)
        if len(out) >= max_tokens:
            break
    return " ".join(out)


def _avoid_profile(avoid: Sequence[str]) -> Tuple[Set[str], List[str]]:
    """Return (tokens to drop from queries, negative terms to reject candidates with).

    A single-concept item ("snow") is also dropped and rejected by its own word.
    A multi-word item ("text on screen") contributes only its expansions, so
    harmless words such as "screen" are not rejected by accident.
    """
    drop: Set[str] = set()
    negatives: List[str] = []
    for raw in avoid or []:
        tokens = _words(str(raw))
        content = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
        single_concept = len(content) <= 1
        for token in tokens:
            key = token if token in AVOID_EXPANSIONS else (token[:-1] if token.endswith("s") else token)
            for term in AVOID_EXPANSIONS.get(key, []):
                drop.add(term)
                if term not in negatives:
                    negatives.append(term)
            if single_concept and token in content and token not in negatives:
                drop.add(token)
                negatives.append(token)
    return drop, negatives


def sanitize_stock_query(raw_query: str, avoid_list: Sequence[str] = ()) -> str:
    """Public v2-compatible sanitiser. Never returns an empty query."""
    drop, _ = _avoid_profile(avoid_list)
    return _sanitize_tokens(raw_query, drop, 4) or "atmospheric cinematic nature"


def _stable_index(text: str, offset: int, modulus: int) -> int:
    """Deterministic index that does not change between Python processes."""
    digest = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    return (digest + offset) % modulus


def _detect_domain(tokens: Sequence[str]) -> str:
    counts = Counter(tokens)
    best, best_score = "general", 0
    for domain, cues in DOMAIN_CUES.items():
        score = sum(counts[cue] for cue in cues)
        if score > best_score:
            best, best_score = domain, score
    return best if best_score >= 2 else "general"


# =====================================================================
# 4. CONTEXT-AWARE CONCEPT ANALYSIS
# =====================================================================

def _resolve_sense(word: str, local: Counter, global_: Counter) -> Tuple[str, float]:
    """Pick the sense whose cues appear most in the local window and the script."""
    default, senses = POLYSEMY_SENSES[word]
    scores: Dict[str, float] = {}
    for name, sense in senses.items():
        local_hits = sum(local.get(cue, 0) for cue in sense.cues)
        global_hits = sum(global_.get(cue, 0) for cue in sense.cues)
        scores[name] = 2.0 * local_hits + 0.5 * min(float(global_hits), 6.0)
    best_name, best_score = max(scores.items(), key=lambda item: (item[1], item[0] == default))
    if best_score <= 0:
        return default, 0.0
    return best_name, best_score


def _sense_confidence(score: float) -> float:
    return 0.35 if score <= 0 else min(0.95, 0.55 + 0.08 * score)


def _trend(tokens: Sequence[str]) -> str:
    up = sum(1 for token in tokens if token in STAT_UP)
    down = sum(1 for token in tokens if token in STAT_DOWN)
    if up == 0 and down == 0:
        return "flat"
    return "up" if up >= down else "down"


def _concept_for(token: str, table: Optional[Dict[str, str]] = None) -> Optional[str]:
    lookup = CONCEPT_VISUALS if table is None else table
    options = [token]
    if token.endswith("s"):
        options.append(token[:-1])
    if token.endswith("es"):
        options.append(token[:-2])
    if token.endswith("ing"):
        options.append(token[:-3])
        options.append(token[:-3] + "e")
    for option in options:
        if option in lookup:
            return lookup[option]
    return None


def _keyword_phrase(tokens: Sequence[str], domain: str, limit: int = 4) -> str:
    """Pick the most visual keywords, preferring concrete nouns and domain cues."""
    cues = DOMAIN_CUES.get(domain, set())
    scored: List[Tuple[float, int, str]] = []
    count = max(len(tokens), 1)
    for position, token in enumerate(tokens):
        if not _is_content_token(token):
            continue
        score = 1.0 - position / count + min(len(token), 10) / 10.0
        if token in CONCRETE_NOUNS:
            score += 3.0
        if token in cues:
            score += 1.5
        scored.append((score, position, token))
    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
    ordered: List[str] = []
    for _, _, token in sorted(top, key=lambda item: item[1]):
        if token not in ordered:
            ordered.append(token)
    return " ".join(ordered)


def _take_all(padded: str, phrase: str) -> Tuple[str, int]:
    """Remove every whole-word occurrence of phrase from a space-padded string."""
    needle = f" {phrase} "
    count = 0
    while needle in padded:
        padded = padded.replace(needle, " ", 1)
        count += 1
    return padded, count


# Priorities: lower numbers come first in the ladder.
PRIORITY_NAMED = 0
PRIORITY_IDIOM = 1
PRIORITY_SENSE = 2
PRIORITY_ABSTRACT = 3
PRIORITY_STAT = 4
PRIORITY_CONCEPT = 5
PRIORITY_GENERIC = 6
PRIORITY_KEYWORD = 7


@dataclass
class _Rules:
    buckets: List[Tuple[int, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    sensitive: bool = False
    stat: bool = False
    abstract: bool = False
    trend: str = "flat"
    keyword: str = ""

    @property
    def concepts(self) -> List[str]:
        ordered: List[str] = []
        for _, visual in sorted(self.buckets, key=lambda bucket: bucket[0]):
            if visual not in ordered:
                ordered.append(visual)
        return ordered


def _analyse_text(
    text_en: str,
    local: Counter,
    global_: Counter,
    domain: str,
    proper: Set[str],
) -> _Rules:
    """Deterministic analysis of one shot's English text. No network, no randomness."""
    rules = _Rules()
    padded = f" {_fold(text_en)} "
    if not padded.strip():
        return rules

    # 1. Named people, brands and lore: replaced, never queried.
    for phrase, visual in NAMED_SORTED:
        padded, count = _take_all(padded, phrase)
        if count:
            rules.buckets.append((PRIORITY_NAMED, visual))
            rules.notes.append(
                f"Named person/brand/lore '{phrase}' detected; neutralized to generic visual '{visual}'."
            )

    tokens = padded.split()
    rest = " " + " ".join(tokens) + " "

    # 2. Idioms.
    for phrase, visual in IDIOM_VISUALS:
        if f" {phrase} " in rest:
            rules.buckets.append((PRIORITY_IDIOM, visual))
            rules.abstract = True
            rules.notes.append(f"Idiom '{phrase}' mapped to physical metaphor '{visual}'.")

    # 3. Polysemous words, resolved from context.
    seen: Set[str] = set()
    for token in tokens:
        if token in POLYSEMY_SENSES and token not in seen:
            seen.add(token)
            name, score = _resolve_sense(token, local, global_)
            sense = POLYSEMY_SENSES[token][1][name]
            rules.buckets.append((PRIORITY_SENSE, sense.visual))
            if sense.sensitive:
                rules.sensitive = True
            rules.notes.append(
                f"Polysemous word '{token}' disambiguated as '{name}' sense "
                f"(confidence {_sense_confidence(score):.2f}) -> '{sense.visual}'."
            )

    # 4. Abstract concepts.
    for token in tokens:
        if token in ABSTRACT_VISUALS:
            rules.buckets.append((PRIORITY_ABSTRACT, ABSTRACT_VISUALS[token]))
            rules.abstract = True
            rules.notes.append(
                f"Abstract concept '{token}' mapped to physical metaphor '{ABSTRACT_VISUALS[token]}'."
            )

    # 5. Statistics, with trend direction.
    if STAT_RE.search(text_en or ""):
        rules.stat = True
        rules.trend = _trend(tokens)
        rules.buckets.append((PRIORITY_STAT, STAT_VISUALS[rules.trend]))
        rules.notes.append(
            f"Numeric statistic detected; mapped to a {TREND_LABEL[rules.trend]} data visual."
        )

    # 6. Concrete concept words. Technology terms count only in a technology script.
    for token in tokens:
        visual = _concept_for(token)
        priority = PRIORITY_CONCEPT
        if visual is None and domain == "technology":
            visual = _concept_for(token, TECH_CONTEXT_VISUALS)
            if visual and token in TECH_GENERIC:
                priority = PRIORITY_GENERIC
        if visual:
            rules.buckets.append((priority, visual))

    # 7. Sensitive vocabulary.
    if any(token in SENSITIVE_TERMS for token in tokens):
        rules.sensitive = True
        rules.notes.append(
            "Sensitive/somber narrative detected; enforced respectful atmospheric visuals and slow pacing."
        )

    # 8. Keyword fallback. Likely names are skipped unless they are known nouns.
    content = [token for token in tokens if token not in proper or token in CONCRETE_NOUNS]
    rules.keyword = _keyword_phrase(content, domain)
    if rules.keyword:
        rules.buckets.append((PRIORITY_KEYWORD, rules.keyword))
    return rules


# =====================================================================
# 5. TIMELINE: BLOCKS AND SHOTS
# =====================================================================

@dataclass
class _Block:
    index: int
    start: float
    end: float
    text: str
    words: List[Dict[str, Any]] = field(default_factory=list)


def _clean_segments(segments: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for seg in segments or []:
        try:
            start = max(0.0, float(seg.get("start", 0.0)))
            end = max(start, float(seg.get("end", start)))
        except (AttributeError, TypeError, ValueError):
            continue
        text = " ".join(str(seg.get("text", "")).split())
        if text:
            cleaned.append({"start": start, "end": end, "text": text, "words": list(seg.get("words") or [])})
    cleaned.sort(key=lambda item: item["start"])
    return cleaned


def _build_blocks(
    segments: Optional[Sequence[Dict[str, Any]]],
    total_duration: float,
    min_dur: float,
    min_last: float,
) -> List[_Block]:
    """Gap-free timeline from 0 to total. Short lines are merged, the tail is absorbed."""
    cleaned = _clean_segments(segments)
    max_end = max((item["end"] for item in cleaned), default=0.0)
    total = max(float(total_duration or 0.0), max_end)
    if total <= 0:
        total = 0.5
    if not cleaned:
        return [_Block(0, 0.0, round(total, 3), "Visual scene")]

    units: List[_Block] = []
    cursor = 0.0
    pending_text: List[str] = []
    pending_words: List[Dict[str, Any]] = []
    for seg in cleaned:
        end = min(max(seg["end"], cursor), total)
        if end - cursor < 1e-6:
            pending_text.append(seg["text"])
            pending_words.extend(seg["words"])
            continue
        text = " ".join(pending_text + [seg["text"]])
        words = pending_words + seg["words"]
        pending_text, pending_words = [], []
        units.append(_Block(0, cursor, end, text, words))
        cursor = end
    if pending_text:
        if units:
            units[-1].text = f"{units[-1].text} {' '.join(pending_text)}"
            units[-1].words.extend(pending_words)
        else:
            units.append(_Block(0, cursor, total, " ".join(pending_text), pending_words))
    if units:
        units[-1].end = total

    blocks: List[_Block] = []
    acc: Optional[_Block] = None
    for unit in units:
        if acc is None:
            acc = _Block(0, unit.start, unit.end, unit.text, list(unit.words))
            continue
        if acc.end - acc.start < min_dur:
            acc.end = unit.end
            acc.text = f"{acc.text} {unit.text}"
            acc.words.extend(unit.words)
        else:
            blocks.append(acc)
            acc = _Block(0, unit.start, unit.end, unit.text, list(unit.words))
    if acc is not None:
        if blocks and acc.end - acc.start < min_last:
            blocks[-1].end = acc.end
            blocks[-1].text = f"{blocks[-1].text} {acc.text}"
            blocks[-1].words.extend(acc.words)
        else:
            blocks.append(acc)
    for index, block in enumerate(blocks):
        block.index = index
    return blocks


def merge_short_segments(segments: List[Dict[str, Any]], total_duration: float) -> List[Dict[str, Any]]:
    """v2-compatible helper: returns merged {start, end, text} blocks."""
    cfg = _planner_config()
    blocks = _build_blocks(segments, total_duration, cfg["min_scene"], cfg["min_last"])
    return [{"start": round(b.start, 3), "end": round(b.end, 3), "text": b.text} for b in blocks]


def _word_start(word: Dict[str, Any], fallback: float) -> float:
    try:
        return float(word.get("start", fallback))
    except (AttributeError, TypeError, ValueError):
        return fallback


def _split_by_words(block: _Block, n: int) -> List[Tuple[float, float, str, List[Dict[str, Any]]]]:
    words = block.words
    duration = block.end - block.start
    cuts: List[int] = []
    previous = 0
    for k in range(1, n):
        target = block.start + duration * k / n
        candidates = range(previous + 1, len(words))
        if len(candidates) == 0:
            break
        best = min(candidates, key=lambda idx: abs(_word_start(words[idx], block.start) - target))
        cuts.append(best)
        previous = best
    bounds = [0] + cuts + [len(words)]
    parts: List[Tuple[float, float, str, List[Dict[str, Any]]]] = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        chunk = words[a:b]
        start = block.start if i == 0 else _word_start(words[a], block.start)
        end = block.end if i == len(bounds) - 2 else _word_start(words[b], block.end)
        text = " ".join(str(w.get("word", w.get("text", ""))).strip() for w in chunk).strip()
        parts.append((start, end, text or block.text, chunk))
    return parts


def _split_block(block: _Block, n: int) -> List[Tuple[float, float, str, List[Dict[str, Any]]]]:
    """Split one block into n contiguous parts, at word or punctuation boundaries."""
    if n <= 1:
        return [(block.start, block.end, block.text, block.words)]
    if block.words and len(block.words) >= n:
        return _split_by_words(block, n)

    tokens = block.text.split()
    n = min(n, len(tokens))
    if n <= 1:
        return [(block.start, block.end, block.text, [])]
    duration = block.end - block.start
    cuts: List[int] = []
    previous = 0
    for k in range(1, n):
        target = round(len(tokens) * k / n)
        best: Optional[int] = None
        for delta in (0, -1, 1, -2, 2):
            j = target + delta
            if previous < j < len(tokens) and re.search(r"[.,;:!?]$", tokens[j - 1]):
                best = j
                break
        if best is None and previous < target < len(tokens):
            best = target
        if best is not None:
            cuts.append(best)
            previous = best
    bounds = [0] + cuts + [len(tokens)]
    total_tokens = len(tokens)
    parts: List[Tuple[float, float, str, List[Dict[str, Any]]]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        start = block.start + duration * a / total_tokens
        end = block.start + duration * b / total_tokens
        parts.append((start, end, " ".join(tokens[a:b]), []))
    first = parts[0]
    parts[0] = (block.start, first[1], first[2], first[3])
    last = parts[-1]
    parts[-1] = (last[0], block.end, last[2], last[3])
    return parts


def _pacing_for(text: str, duration: float) -> str:
    if duration <= 0:
        return "medium"
    words_per_second = len(text.split()) / duration
    if words_per_second >= 3.5:
        return "fast"
    if words_per_second <= 1.6:
        return "slow"
    return "medium"


def _hint_text(scene: Any) -> str:
    if scene is None:
        return ""
    if isinstance(scene, dict):
        value = scene.get("visual_hint")
    else:
        value = getattr(scene, "visual_hint", None)
    return " ".join(str(value).split()) if value else ""


def _hints_by_block(explicit_scenes: Optional[Sequence[Any]], count: int) -> List[str]:
    hints: List[str] = []
    for i in range(count):
        if explicit_scenes and i < len(explicit_scenes):
            hints.append(_hint_text(explicit_scenes[i]))
        else:
            hints.append("")
    return hints


# =====================================================================
# 6. DRAFT SHOTS AND CONTEXT
# =====================================================================

@dataclass
class _Draft:
    index: int = 0
    block_index: int = 0
    shot_index: int = 0
    shots_in_block: int = 1
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    text_en: str = ""
    language: str = "en"
    pacing: str = "medium"
    hint: str = ""
    hint_en: str = ""
    sensitive: bool = False
    sensitive_rule: bool = False
    stat: bool = False
    abstract: bool = False
    trend: str = "flat"
    keyword: str = ""
    primary: str = ""
    secondary: str = ""
    fallback: str = ""
    ladder: List[str] = field(default_factory=list)
    alternates: List[str] = field(default_factory=list)
    anchors: List[str] = field(default_factory=list)
    negatives: List[str] = field(default_factory=list)
    variant_index: int = 0
    crop: float = 0.5
    shot_type: str = ""
    camera_movement: str = ""
    lighting_style: str = ""
    color_grading: str = ""
    emotional_tone: str = ""
    physical_translation: str = ""
    source: str = "rules"
    confidence: float = 0.5
    notes: List[str] = field(default_factory=list)


def _build_drafts(blocks: Sequence[_Block], hints: Sequence[str], cfg: Dict[str, Any]) -> List[_Draft]:
    drafts: List[_Draft] = []
    for block in blocks:
        duration = block.end - block.start
        pacing = _pacing_for(block.text, duration)
        max_shot = cfg["max_shot"]
        if max_shot > 0:
            if any(token in SENSITIVE_TERMS for token in _words(block.text)):
                max_shot *= 1.5
            if pacing == "fast":
                max_shot *= 0.75
        count = 1
        if max_shot > 0 and duration > max_shot:
            count = math.ceil(duration / max_shot - 1e-9)
        count = max(1, min(count, int(duration // max(0.8 * cfg["min_scene"], 1.0))))
        parts = _split_block(block, count)
        hint = hints[block.index] if block.index < len(hints) else ""
        for shot_index, (start, end, text, _part_words) in enumerate(parts):
            drafts.append(_Draft(
                block_index=block.index,
                shot_index=shot_index,
                shots_in_block=len(parts),
                start=start,
                end=end,
                text=text,
                pacing=pacing,
                hint=hint,
            ))
    for index, draft in enumerate(drafts):
        draft.index = index
    return drafts


@dataclass
class _Context:
    style: str
    avoid: List[str]
    drop: Set[str]
    negatives: List[str]
    domain: str
    global_counts: Counter
    proper: Set[str]
    palette: str = ""
    brief: Any = None


def _attach_english(drafts: Sequence[_Draft]) -> None:
    for draft in drafts:
        draft.language = _detect_language(draft.text)
        if draft.language == "en":
            draft.text_en = draft.text
        else:
            gloss, hits = _lexicon_gloss(draft.text)
            latin = " ".join(_words(draft.text))
            draft.text_en = " ".join(part for part in (gloss, latin) if part).strip()
            if hits == 0 and not latin:
                draft.notes.append(
                    "Non-English narration with no offline lexicon match; a generic visual was used."
                )
        if draft.hint:
            if _detect_language(draft.hint) == "en":
                draft.hint_en = draft.hint
            else:
                gloss, _ = _lexicon_gloss(draft.hint)
                draft.hint_en = gloss or draft.hint


def _off_topic_negatives(tokens: Sequence[str], domain: str) -> List[str]:
    """Subjects a clip must not show, unless the script itself is about them."""
    present = set(tokens)
    found: List[str] = []
    if not present & SPACE_WORDS:
        found.extend(SPACE_NEGATIVES)
    if domain != "sports" and not present & SPORT_WORDS:
        found.extend(SPORT_NEGATIVES)
    return found


def _build_context(drafts: Sequence[_Draft], full_transcript: str, style: str, avoid: List[str]) -> _Context:
    drop, negatives = _avoid_profile(avoid)
    tokens: List[str] = []
    for draft in drafts:
        tokens.extend(_words(draft.text_en))
    domain = _detect_domain(tokens)
    return _Context(
        style=style,
        avoid=list(avoid),
        drop=drop,
        negatives=_dedupe(list(negatives) + _off_topic_negatives(tokens, domain)),
        domain=domain,
        global_counts=Counter(tokens),
        proper=_proper_nouns(full_transcript),
    )


def _window_counts(drafts: Sequence[_Draft], index: int) -> Counter:
    counts: Counter = Counter()
    for j in range(max(0, index - 1), min(len(drafts), index + 2)):
        counts.update(_words(drafts[j].text_en))
    return counts


def _crop_from_text(text_en: str, hint: str, index: int) -> float:
    combined = f"{_fold(text_en)} {_fold(hint)}"
    if re.search(r"\b(left edge|far left|on the left|left side)\b", combined):
        return 0.25
    if re.search(r"\b(right edge|far right|on the right|right side|at the edge)\b", combined):
        return 0.75
    if "rule of thirds" in combined:
        return 0.35 if index % 2 == 0 else 0.65
    return 0.5


def _ladder_parts(phrases: Sequence[str], domain: str, drop: Set[str]) -> Tuple[List[str], List[str]]:
    """Return (specific queries, broad queries). Specific ones come first."""
    specific = list(phrases)
    if specific:
        base = specific[0].split()
        qualifiers = QUAL_PLACE if any(token in OUTDOOR_TOKENS for token in base) else QUAL_OBJECT
        for qualifier in qualifiers:
            if qualifier in base:
                continue
            variant = " ".join((base[:3] + [qualifier])[:4])
            if variant not in specific:
                specific.append(variant)
    pool = DOMAIN_BROAD_POOLS.get(domain, DOMAIN_BROAD_POOLS["general"])
    broad = _dedupe(_sanitize_tokens(item, drop, 4) for item in pool)
    return specific, broad


# =====================================================================
# 7. RULE-BASED FILL, CINEMATOGRAPHY AND VARIETY
# =====================================================================

def _fill_from_rules(draft: _Draft, rules: _Rules, ctx: _Context) -> None:
    draft.sensitive = rules.sensitive
    draft.sensitive_rule = rules.sensitive
    draft.stat = rules.stat
    draft.abstract = rules.abstract
    draft.trend = rules.trend
    draft.keyword = rules.keyword
    draft.notes.extend(rules.notes)

    drop = ctx.drop
    concepts: List[str] = []
    if draft.hint_en:
        concepts.append(draft.hint_en)
    if draft.sensitive:
        # Safety: a sensitive shot never queries the raw wording, only somber visuals.
        drop = ctx.drop | SENSITIVE_TERMS
        start = _stable_index(draft.text, draft.shot_index, len(SENSITIVE_POOL))
        concepts.extend(SENSITIVE_POOL[(start + k) % len(SENSITIVE_POOL)] for k in range(3))
        concepts.extend(item for item in rules.concepts if item != rules.keyword)
    else:
        concepts.extend(rules.concepts)

    phrases = _dedupe(_sanitize_tokens(item, drop, 4) for item in concepts)
    specific, broad = _ladder_parts(phrases, ctx.domain, drop)
    if not specific and draft.language != "en" and not draft.sensitive:
        default = MULTILINGUAL_DEFAULTS[draft.index % len(MULTILINGUAL_DEFAULTS)]
        specific = _dedupe([_sanitize_tokens(default, ctx.drop, 4)])
    if not specific:
        specific = broad[:1] or ["atmospheric landscape"]

    ladder = _dedupe(specific + broad)
    draft.ladder = ladder
    draft.primary = ladder[0]
    draft.alternates = [q for q in specific[1:] if q != draft.primary][:4]
    draft.secondary = next((q for q in specific[1:] if q != draft.primary), broad[0] if broad else draft.primary)
    draft.fallback = broad[0] if broad else draft.primary
    draft.anchors = draft.primary.split()[:2]
    draft.negatives = _dedupe(ctx.negatives + (SENSITIVE_NEGATIVES if draft.sensitive else []))
    draft.crop = _crop_from_text(draft.text_en, draft.hint_en, draft.index)
    draft.physical_translation = draft.primary
    if draft.hint_en:
        draft.source = "explicit_hint"
        draft.confidence = 0.95
    else:
        draft.source = "rules"
        draft.confidence = 0.6 if rules.concepts else 0.3


PLACE_SHOTS: List[str] = [
    "Wide Establishing Shot", "Extreme Wide Aerial Shot", "Medium Shot",
    "Low-Angle Medium Shot", "Tracking Side Shot",
]
OBJECT_SHOTS: List[str] = [
    "Close-Up", "Insert Detail", "Overhead Top-Down Shot", "Medium Shot",
]


def _next_shot_type(index: int, previous: str, sensitive: bool, primary: str = "") -> str:
    """Rotate shot sizes, choosing the family that suits the subject."""
    if index == 0:
        return "Static Wide Shot" if sensitive else "Wide Establishing Shot"
    if sensitive:
        options = ["Static Wide Shot", "Static Medium Shot"]
    elif any(token in PLACE_TOKENS for token in primary.split()):
        options = PLACE_SHOTS
    else:
        options = OBJECT_SHOTS
    start = index % len(options)
    ordered = options[start:] + options[:start]
    for option in ordered:
        if option != previous:
            return option
    return ordered[0]


def _camera_for(shot_type: str, pacing: str, stat: bool, sensitive: bool) -> str:
    lowered = shot_type.lower()
    if sensitive:
        return "Locked-Off Tripod"
    if pacing == "fast":
        return "Dynamic Handheld Tracking"
    if stat:
        return "Slow Push-In"
    if "aerial" in lowered:
        return "Slow Drone Glide"
    if "close" in lowered or "detail" in lowered or "macro" in lowered:
        return "Slow Rack Focus"
    return "Slow Smooth Gimbal Push-In"


def _lighting_for(text: str, sensitive: bool) -> str:
    if sensitive:
        return "Soft Diffused Overcast Light"
    lowered = text.lower()
    if any(word in lowered for word in ("morning", "sunrise", "sunlight", "dawn", "golden")):
        return "Golden Hour Backlight"
    if any(word in lowered for word in ("night", "candle", "darkness", "bedside", "lamp", "dark")):
        return "Warm Low-Key Practical Lighting"
    if any(word in lowered for word in ("office", "laptop", "screen", "chart", "analyst", "desk")):
        return "Crisp Daylight"
    return "Natural Cinematic Contrast"


def _tone_for(sensitive: bool, stat: bool, abstract: bool, pacing: str) -> str:
    if sensitive:
        return "Somber & Respectful"
    if stat:
        return "Confident & Analytical"
    if abstract:
        return "Inspirational"
    if pacing == "fast":
        return "Energetic"
    return "Engaging"


def _apply_diversity(drafts: Sequence[_Draft]) -> None:
    """Stop nearby shots from repeating the same stock search."""
    recent: deque = deque(maxlen=3)
    counts: Counter = Counter()
    for draft in drafts:
        options = _dedupe([draft.primary] + draft.ladder)
        chosen = next((q for q in options if q not in recent and counts[q] < 2), options[0])
        if chosen != draft.primary:
            draft.notes.append(
                f"Query varied to '{chosen}' so nearby shots do not repeat the same stock search."
            )
        draft.primary = chosen
        draft.ladder = _dedupe([chosen] + draft.ladder)
        draft.variant_index = counts[chosen]
        counts[chosen] += 1
        recent.append(chosen)


def _enforce_cinematography(drafts: Sequence[_Draft], ctx: _Context) -> None:
    """Final cinematography rules, applied after any LLM output."""
    previous = ""
    for draft in drafts:
        shot = draft.shot_type or _next_shot_type(draft.index, previous, draft.sensitive, draft.primary)
        if draft.sensitive and "static" not in shot.lower():
            shot = "Static Wide Shot" if ("wide" in shot.lower() or "aerial" in shot.lower()) else "Static Medium Shot"
        if draft.index == 0 and not draft.sensitive:
            if "wide" not in shot.lower() and "aerial" not in shot.lower():
                shot = "Wide Establishing Shot"
        if shot == previous:
            shot = _next_shot_type(draft.index + 1, previous, draft.sensitive, draft.primary)
        draft.shot_type = shot
        previous = shot

        if draft.sensitive:
            draft.camera_movement = "Locked-Off Tripod"
            draft.lighting_style = "Soft Diffused Overcast Light"
            draft.color_grading = "Muted Respectful Somber"
            draft.emotional_tone = "Somber & Respectful"
        else:
            draft.camera_movement = draft.camera_movement or _camera_for(shot, draft.pacing, draft.stat, False)
            draft.lighting_style = draft.lighting_style or _lighting_for(draft.primary, False)
            draft.color_grading = ctx.palette or draft.color_grading or "Warm Film Grade"
            draft.emotional_tone = draft.emotional_tone or _tone_for(False, draft.stat, draft.abstract, draft.pacing)


# =====================================================================
# 8. OUTPUT SHAPING
# =====================================================================

def _draft_to_scene(draft: _Draft) -> Dict[str, Any]:
    return {
        "scene_index": draft.index,
        "block_index": draft.block_index,
        "shot_index": draft.shot_index,
        "shots_in_block": draft.shots_in_block,
        "start_time": round(draft.start, 3),
        "end_time": round(draft.end, 3),
        "duration": round(draft.end - draft.start, 3),
        "narration_text": draft.text,
        "visual_query": draft.primary,
        "secondary_query": draft.secondary,
        "fallback_query": draft.fallback,
        "query_ladder": list(draft.ladder),
        "alternate_queries": list(draft.alternates),
        "variant_index": draft.variant_index,
        "must_have_terms": list(draft.anchors),
        "negative_terms": list(draft.negatives),
        "is_abstract_or_stat": bool(draft.stat or draft.abstract),
        "is_sensitive": bool(draft.sensitive),
        "language": draft.language,
        "pacing": draft.pacing,
        "crop_x_center": round(max(0.15, min(0.85, float(draft.crop))), 2),
        "resolution_source": draft.source,
        "confidence": round(float(draft.confidence), 2),
        "cinematography": {
            "shot_type": draft.shot_type,
            "camera_movement": draft.camera_movement,
            "lighting_style": draft.lighting_style,
            "color_grading": draft.color_grading,
            "emotional_tone": draft.emotional_tone,
            "physical_translation": draft.physical_translation or draft.primary,
        },
    }


def _draft_warnings(draft: _Draft) -> List[str]:
    messages = [f"Scene {draft.index}: {note}" for note in _dedupe(draft.notes)]
    if draft.shots_in_block > 1 and draft.shot_index == 0:
        messages.append(
            f"Scene {draft.index}: Long narration line split into {draft.shots_in_block} shots "
            f"to keep B-roll cuts at a natural pace."
        )
    return messages


# =====================================================================
# 9. GEMINI SCHEMAS
# =====================================================================

class GlossaryEntry(BaseModel):
    term: str = Field(description="A word or name in the script that needs one consistent visual meaning.")
    meaning_in_context: str = Field(description="What the term means in this script, in English.")
    visual_guidance: str = Field(description="A concrete, filmable real-world visual for the term. No names, no brands.")


class Stage0GlobalBrief(BaseModel):
    overarching_narrative_theme: str = Field(description="Global summary of the script's setting, genre, audience and visual arc.")
    visual_pacing_strategy: str = Field(description="How camera energy and mood should evolve across the whole video.")
    domain: str = Field(description="One of: technology, finance, nature, health, travel, food, education, sports, fantasy, sensitive, lifestyle, business, general.")
    is_sensitive_topic: bool = Field(description="True if the script is about tragedy, death, disaster, grief, war or violence.")
    color_palette: str = Field(description="One colour grade for the whole video, for example 'Teal and Amber Cinematic'.")
    glossary: List[GlossaryEntry] = Field(default_factory=list, description="Polysemous words, idioms, fictional names, brands, real people and foreign terms, each with one visual.")
    banned_visual_concepts: List[str] = Field(default_factory=list, description="Concepts that must never appear anywhere in this video.")


class Stage1SceneDeconstruction(BaseModel):
    index: int = Field(description="The exact shot index from the input list.")
    english_translation: str = Field(default_factory=str, description="The narration rendered in English. Copy it if it is already English.")
    emotional_tone: str = Field(default_factory=str, description="The mood and emotional subtext of this specific line.")
    abstract_or_lore_terms_detected: List[str] = Field(default_factory=list, description="Idioms, brand names, real names, polysemous words, metaphors or abstract concepts in the line.")
    disambiguations: List[str] = Field(default_factory=list, description="Each resolved word written as 'word -> meaning in this context'.")
    physical_world_translation: str = Field(description="A 100% real-world, physical description a real camera could film today.")
    subject_anchor: str = Field(description="The core physical noun that must anchor the frame, for example 'gothic castle' or 'server rack'.")
    is_abstract_or_stat: bool = Field(description="True if the line is a numeric statistic, percentage or proverb.")
    is_sensitive: bool = Field(default_factory=bool, description="True if the line concerns tragedy, death, disaster, grief, war or violence.")


class Stage1DeconstructionResponse(BaseModel):
    overarching_narrative_theme: str = Field(default_factory=str, description="Global summary of the script's setting and visual arc.")
    visual_pacing_strategy: str = Field(default_factory=str, description="How the camera energy and mood should evolve across these shots.")
    scenes: List[Stage1SceneDeconstruction] = Field(default_factory=list)


class Stage2CinematicDirective(BaseModel):
    index: int = Field(description="The exact shot index.")
    shot_type: str = Field(default_factory=str, description="Framing, for example 'Extreme Wide Aerial Shot', 'Macro Close-Up', 'Low-Angle Medium Shot'.")
    camera_movement: str = Field(default_factory=str, description="Movement, for example 'Slow Push-In', 'Smooth Gimbal Tracking', 'Locked-Off Tripod'.")
    lighting_style: str = Field(default_factory=str, description="Lighting, for example 'Golden Hour Backlight', 'Moody Volumetric Fog', 'Crisp Daylight'.")
    color_grading: str = Field(default_factory=str, description="Colour palette, for example 'Teal and Amber Cinematic'.")
    primary_stock_query: str = Field(description="3 or 4 concrete English nouns or adjectives for Pexels and Pixabay. No camera terms, punctuation, prepositions, names, brands or fictional names. Never abstract words such as future, technology, goal or workflow.")
    alternate_queries: List[str] = Field(default_factory=list, description="Exactly three different queries showing the same idea from other angles or settings. 3 or 4 words each.")
    secondary_stock_query: str = Field(default_factory=str, description="A 2 or 3 word query, broader than the primary.")
    fallback_broad_query: str = Field(default_factory=str, description="A 2 word atmospheric query with thousands of stock hits.")
    crop_x_center: float = Field(default=0.5, description="Horizontal focus for 9:16 cropping: 0.25 if the subject is on the left, 0.75 if on the right, otherwise 0.5.")


class Stage2CinematographyResponse(BaseModel):
    directives: List[Stage2CinematicDirective] = Field(default_factory=list)


PLANNER_SYSTEM_PROMPT = """
You are an expert AI Cinematographer and B-Roll Director. You build a timed sequence of
stock-footage shots for a narrated video. Every shot must be something a real camera could film
today, and every stock query must work on Pexels and Pixabay.

NON-NEGOTIABLE RULES
1. Context disambiguation. Never match keywords naively. Resolve words with several meanings from
   the surrounding paragraph and the whole script.
   - 'Python' in software -> programmer coding on a laptop. Never a snake unless the script is about wildlife.
   - 'Bank' in finance -> modern bank or financial district. Never a riverbank unless the script is about water.
   - 'Cloud' in servers -> data-centre server racks. Never sky clouds unless the script is about weather.
   - 'Mouse' in computing -> computer mouse on a desk. Never a rodent unless the script is about animals.
2. Real people and brands. Never output a real named person, celebrity, politician, brand or logo in a
   visual description or query. Replace each with a generic equivalent, for example 'tech executive on
   conference stage' or 'unbranded sports sneakers'.
3. Abstract ideas and numbers. Turn proverbs, idioms and abstract nouns into physical metaphors
   ('trust is earned slowly' -> 'two hands shaking in agreement'). Turn statistics into data visuals
   ('sales grew 40%' -> 'analyst pointing at rising bar chart on monitor').
4. Sensitive material. For tragedy, death, disaster, grief, war or violence, show respectful, somber,
   static environmental visuals such as a candle in darkness, rain on a window or an empty park bench.
   Never depict injury, bodies, blood, weapons, wreckage or identifiable victims.
5. Language. Narration may be in Tamil, Hindi or another language. Reason in English and write every
   query in English.
6. Stock queries. Use 3 to 4 concrete English nouns or adjectives. No camera jargon, punctuation,
   prepositions, fictional names, brand names or people's names. Describe what is physically visible.
7. Variety. Neighbouring shots must not repeat a subject. Vary angles (wide, medium, close-up, detail,
   aerial, overhead) and settings.
8. Respect explicit editor hints and the list of things to avoid. Never show anything on the avoid list.
9. Film the thing, not the idea. Software, technology and business narration is abstract. Show the
   tool, place or action that stands for it: 'automation' -> 'robotic arm working in factory';
   'developers' -> 'software developer coding laptop'; 'server-side processing' -> 'server room racks
   blinking lights'; 'architecture' -> 'engineers drawing diagram whiteboard'; 'workflows' -> 'sticky
   notes planning board'; 'rendering' -> 'editing software timeline'. Never put these words in a
   query: future, technology, goal, automation, architecture, systems, streamline, workflow, pipeline,
   seamless, intelligent, clean, performance, complex, stunning, ideas, reality. A stock search reads
   'goal' as a football goal and 'future' as space imagery.
Return only JSON that matches the requested schema.
"""


# =====================================================================
# 10. GEMINI CLIENT, RETRIES AND PARSING
# =====================================================================

_TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504", "resource_exhausted", "unavailable", "deadline",
    "timed out", "timeout", "internal", "temporarily", "connection",
)


def _is_transient_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _with_retries(call, retries: int, label: str):
    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:
            if attempt >= retries or not _is_transient_error(exc):
                raise
            delay = min(30.0, 2.0 ** attempt + random.uniform(0.0, 1.0))
            log.warning(
                "%s: transient error (attempt %d/%d), retrying in %.1fs: %s",
                label, attempt + 1, retries, delay, str(exc)[:160],
            )
            time.sleep(delay)
            attempt += 1


def _make_genai_client(api_key: str, timeout_ms: int):
    from google import genai
    from google.genai import types
    try:
        return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))
    except Exception:
        return genai.Client(api_key=api_key)


def _strip_fences(text: str) -> str:
    cleaned = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fenced:
        return fenced.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def _parse_structured(schema, text: str):
    cleaned = _strip_fences(text)
    try:
        return schema.model_validate_json(cleaned)
    except Exception:
        return schema.model_validate(json.loads(cleaned))


def _generate_json(client, cfg: Dict[str, Any], prompt: str, schema, temperature: float, label: str):
    from google.genai import types
    config = types.GenerateContentConfig(
        system_instruction=PLANNER_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
    )
    last_error: Optional[Exception] = None
    for parse_attempt in range(2):
        response = _with_retries(
            lambda: client.models.generate_content(model=cfg["model"], contents=prompt, config=config),
            cfg["retries"],
            label,
        )
        try:
            return _parse_structured(schema, getattr(response, "text", "") or "")
        except Exception as exc:
            last_error = exc
            log.warning("%s: unparseable structured output (attempt %d): %s",
                        label, parse_attempt + 1, str(exc)[:160])
    raise RuntimeError(f"{label}: could not parse model output ({str(last_error)[:120]})")


def _brief_summary(brief: Stage0GlobalBrief) -> str:
    lines = [
        f"Theme: {brief.overarching_narrative_theme}",
        f"Pacing strategy: {brief.visual_pacing_strategy}",
        f"Domain: {brief.domain}",
        f"Sensitive topic: {brief.is_sensitive_topic}",
        f"Colour palette (use for every shot): {brief.color_palette}",
    ]
    if brief.glossary:
        lines.append("Glossary (apply consistently):")
        lines.extend(
            f"- {entry.term}: {entry.meaning_in_context} -> visual: {entry.visual_guidance}"
            for entry in brief.glossary[:40]
        )
    if brief.banned_visual_concepts:
        lines.append("Never show: " + ", ".join(brief.banned_visual_concepts[:30]))
    return "\n".join(lines)


# =====================================================================
# 11. GEMINI STAGES
# =====================================================================

def _call_stage0(client, cfg, full_transcript: str, style: str, avoid: List[str], hints: List[Dict[str, Any]]):
    prompt = (
        "GLOBAL SCRIPT (original wording and casing):\n"
        f'"""\n{full_transcript}\n"""\n\n'
        f"DESIRED VISUAL STYLE: {style}\n"
        f"STRICTLY AVOID SHOWING: {json.dumps(avoid, ensure_ascii=False)}\n"
        f"EDITOR VISUAL HINTS (anchor these lines to the hinted visual): {json.dumps(hints, ensure_ascii=False)}\n\n"
        "TASK (STAGE 0 - GLOBAL VISUAL BRIEF):\n"
        "1. Identify the domain, the tone and whether the topic is sensitive.\n"
        "2. Build a glossary of every word or name that needs one consistent visual meaning across the whole "
        "video: polysemous words, idioms, fictional names, brands, real people and foreign-language terms.\n"
        "3. List concepts that must never appear anywhere in this video.\n"
        "4. Choose one colour palette for the whole video.\n"
    )
    return _generate_json(client, cfg, prompt, Stage0GlobalBrief, 0.2, "stage0")


def _call_stage1(client, cfg, brief_text: str, style: str, avoid: List[str], items: List[Dict[str, Any]]):
    prompt = (
        f"GLOBAL VISUAL BRIEF:\n{brief_text}\n\n"
        f"DESIRED VISUAL STYLE: {style}\n"
        f"STRICTLY AVOID SHOWING: {json.dumps(avoid, ensure_ascii=False)}\n\n"
        "TASK (STAGE 1 - SEMANTIC DECONSTRUCTION). For every shot below:\n"
        "1. english_translation: render the narration in English (copy it if it is already English).\n"
        "2. Resolve each polysemous word, idiom, metaphor and name using the line, its neighbouring shots and "
        "the glossary. Record each in disambiguations as 'word -> meaning in this context'.\n"
        "3. Neutralise real people and brands into generic, unbranded real-world equivalents.\n"
        "4. Translate abstract lines, proverbs and statistics into concrete physical images.\n"
        "5. is_sensitive: true for tragedy, death, disaster, grief, war or violence. Keep such visuals respectful and non-graphic.\n"
        "6. physical_world_translation must describe something a real camera could film today.\n\n"
        f"SHOTS:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n"
    )
    return _generate_json(client, cfg, prompt, Stage1DeconstructionResponse, 0.2, "stage1")


def _call_stage2(client, cfg, brief_text: str, style: str, avoid: List[str], items: List[Dict[str, Any]]):
    prompt = (
        f"GLOBAL VISUAL BRIEF:\n{brief_text}\n\n"
        f"DESIRED VISUAL STYLE: {style}\n"
        f"STRICTLY AVOID: {json.dumps(avoid, ensure_ascii=False)}\n\n"
        "TASK (STAGE 2 - CINEMATOGRAPHY AND STOCK SEARCH). For every shot:\n"
        "1. Choose shot_type, camera_movement, lighting_style and color_grading. Use the palette from the brief. "
        "Vary shot sizes between neighbouring shots.\n"
        "2. primary_stock_query: 3 or 4 concrete English nouns or adjectives. No camera terms, punctuation, "
        "prepositions, names, brands or fictional names. Describe what is physically visible.\n"
        "3. alternate_queries: exactly 3 different queries showing the same idea from other angles or settings.\n"
        "4. secondary_stock_query: 2 or 3 words, broader than the primary.\n"
        "5. fallback_broad_query: 2 atmospheric words with thousands of stock hits.\n"
        "6. crop_x_center: 0.25 if the subject sits on the left, 0.75 if on the right, otherwise 0.5.\n\n"
        f"SHOTS WITH STAGE 1 TRANSLATIONS:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n"
    )
    return _generate_json(client, cfg, prompt, Stage2CinematographyResponse, 0.15, "stage2")


def _process_chunk(client, cfg, brief, style: str, avoid: List[str], chunk: List[Dict[str, Any]], deadline: float):
    if time.monotonic() > deadline:
        raise TimeoutError("planner time budget exhausted")
    brief_text = _brief_summary(brief)
    s1 = _call_stage1(client, cfg, brief_text, style, avoid, chunk)
    s1_map = {int(item.index): item for item in s1.scenes}
    stage2_items: List[Dict[str, Any]] = []
    for item in chunk:
        translated = s1_map.get(item["index"])
        stage2_items.append({
            **item,
            "english": translated.english_translation if translated else "",
            "physical": translated.physical_world_translation if translated else item["narration"],
            "subject_anchor": translated.subject_anchor if translated else "",
            "sensitive": bool(translated.is_sensitive) if translated else False,
        })
    s2 = _call_stage2(client, cfg, brief_text, style, avoid, stage2_items)
    s2_map = {int(item.index): item for item in s2.directives}
    return s1_map, s2_map


class _LLMState:
    """Thread-safe progress record, so partial results survive a timeout."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.brief: Any = None
        self.stage1: Dict[int, Any] = {}
        self.stage2: Dict[int, Any] = {}
        self.errors: List[str] = []

    def record(self, s1_map: Dict[int, Any], s2_map: Dict[int, Any]) -> None:
        with self.lock:
            self.stage1.update(s1_map)
            self.stage2.update(s2_map)

    def error(self, message: str) -> None:
        with self.lock:
            self.errors.append(message)

    def snapshot(self):
        with self.lock:
            return self.brief, dict(self.stage1), dict(self.stage2), list(self.errors)


def _run_llm_pipeline(
    state: _LLMState,
    payload: List[Dict[str, Any]],
    full_transcript: str,
    style: str,
    avoid: List[str],
    cfg: Dict[str, Any],
    deadline: float,
) -> None:
    """Synchronous pipeline. Runs in a worker thread via asyncio.to_thread."""
    client = _make_genai_client(settings.GEMINI_API_KEY, cfg["timeout_ms"])
    hints = [{"line": item["index"], "hint": item["visual_hint"]} for item in payload if item["visual_hint"]]
    brief = _call_stage0(client, cfg, full_transcript, style, avoid, hints)
    with state.lock:
        state.brief = brief

    size = cfg["chunk_size"]
    chunks = [payload[i:i + size] for i in range(0, len(payload), size)]
    with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
        futures = {
            pool.submit(_process_chunk, client, cfg, brief, style, avoid, chunk, deadline): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            chunk = futures[future]
            label = f"shots {chunk[0]['index']}-{chunk[-1]['index']}"
            try:
                s1_map, s2_map = future.result()
                state.record(s1_map, s2_map)
            except Exception as exc:
                log.warning("planner %s failed: %s", label, str(exc)[:200])
                state.error(f"{label} fell back to rule-based planning ({str(exc)[:100]})")


@dataclass
class _LLMOutcome:
    brief: Any
    stage1: Dict[int, Any]
    stage2: Dict[int, Any]
    errors: List[str]


_CACHE: "OrderedDict[str, _LLMOutcome]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _cache_get(key: str) -> Optional[_LLMOutcome]:
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    return None


def _cache_put(key: str, value: _LLMOutcome, size: int) -> None:
    if size <= 0:
        return
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > size:
            _CACHE.popitem(last=False)


def _cache_key(payload: List[Dict[str, Any]], style: str, avoid: List[str], cfg: Dict[str, Any]) -> str:
    blob = json.dumps(
        {"v": PLANNER_VERSION, "model": cfg["model"], "style": style, "avoid": avoid, "shots": payload},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


async def _run_llm_with_budget(
    payload: List[Dict[str, Any]],
    full_transcript: str,
    style: str,
    avoid: List[str],
    cfg: Dict[str, Any],
) -> _LLMOutcome:
    key = _cache_key(payload, style, avoid, cfg)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    state = _LLMState()
    deadline = time.monotonic() + cfg["budget_sec"]
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_run_llm_pipeline, state, payload, full_transcript, style, avoid, cfg, deadline),
            timeout=cfg["budget_sec"] + 60,
        )
    except asyncio.TimeoutError:
        state.error("planner time budget exceeded; remaining shots use rule-based planning")

    brief, stage1, stage2, errors = state.snapshot()
    outcome = _LLMOutcome(brief=brief, stage1=stage1, stage2=stage2, errors=errors)
    if brief is not None and not errors and len(stage2) >= len(payload):
        _cache_put(key, outcome, cfg["cache_size"])
    return outcome


# =====================================================================
# 12. MERGING GEMINI OUTPUT
# =====================================================================

def _apply_llm(draft: _Draft, s1: Any, s2: Any, ctx: _Context) -> None:
    anchors: List[str] = []
    if s1 is not None:
        if s1.english_translation and draft.language != "en":
            draft.text_en = _clean_text(s1.english_translation)
        if s1.physical_world_translation:
            draft.physical_translation = _clean_text(s1.physical_world_translation)
        if s1.emotional_tone:
            draft.emotional_tone = _clean_text(s1.emotional_tone)
        if s1.is_sensitive:
            draft.sensitive = True
        if s1.is_abstract_or_stat:
            draft.abstract = True
        for item in list(s1.disambiguations)[:6]:
            draft.notes.append(f"Disambiguated: {_clean_text(item)}")
        anchors = _sanitize_tokens(s1.subject_anchor, ctx.drop, 2).split()
    if s2 is None:
        if anchors:
            draft.anchors = anchors
        return

    queries = [
        query for query in _dedupe(
            _sanitize_tokens(query, ctx.drop, 4)
            for query in [s2.primary_stock_query, *list(s2.alternate_queries),
                          s2.secondary_stock_query, s2.fallback_broad_query]
        )
        if _has_subject(query)
    ]
    if not queries:
        if anchors:
            draft.anchors = anchors
        return

    draft.source = "llm"
    draft.confidence = max(draft.confidence, 0.85)
    if draft.hint_en or draft.sensitive_rule:
        # Editor hints and rule-based safety pools keep the primary slot.
        ladder = [draft.primary] + queries + draft.ladder[1:]
    else:
        ladder = queries + draft.ladder
    draft.ladder = _dedupe(ladder)
    draft.primary = draft.ladder[0]
    draft.alternates = [q for q in queries[1:] if q != draft.primary][:4]

    llm_secondary = _sanitize_tokens(s2.secondary_stock_query, ctx.drop, 4)
    llm_fallback = _sanitize_tokens(s2.fallback_broad_query, ctx.drop, 4)
    if llm_secondary:
        draft.secondary = llm_secondary
    if llm_fallback:
        draft.fallback = llm_fallback

    draft.anchors = anchors or draft.primary.split()[:2]
    shot = _clean_text(s2.shot_type)
    if shot:
        draft.shot_type = shot
    camera = _clean_text(s2.camera_movement)
    if camera:
        draft.camera_movement = camera
    lighting = _clean_text(s2.lighting_style)
    if lighting:
        draft.lighting_style = lighting
    grading = _clean_text(s2.color_grading)
    if grading:
        draft.color_grading = grading
    if abs(float(s2.crop_x_center) - 0.5) > 1e-6:
        draft.crop = float(s2.crop_x_center)


def _llm_payload(drafts: Sequence[_Draft]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for position, draft in enumerate(drafts):
        before = drafts[position - 1].text if position > 0 else ""
        after = drafts[position + 1].text if position + 1 < len(drafts) else ""
        payload.append({
            "index": draft.index,
            "narration": draft.text,
            "visual_hint": draft.hint,
            "position": f"shot {draft.shot_index + 1} of {draft.shots_in_block} from this narration line",
            "duration_sec": round(draft.end - draft.start, 2),
            "pacing": draft.pacing,
            "context_before": before[-200:],
            "context_after": after[:200],
        })
    return payload


# =====================================================================
# 13. MAIN ENTRY POINT (called by the worker pipeline)
# =====================================================================

async def plan_scenes(
    segments: List[Dict[str, Any]],
    total_duration: float,
    style: str,
    avoid: List[str],
    explicit_scenes: Optional[List[SceneInput]] = None,
) -> Tuple[List[Dict[str, Any]], List[str], str]:
    cfg = _planner_config()
    style = _clean_text(style)
    avoid = [_clean_text(item) for item in (avoid or []) if _clean_text(item)]
    warnings: List[str] = []

    blocks = _build_blocks(segments, total_duration, cfg["min_scene"], cfg["min_last"])
    hints = _hints_by_block(explicit_scenes, len(blocks))
    drafts = _build_drafts(blocks, hints, cfg)
    _attach_english(drafts)
    full_transcript = " ".join(block.text for block in blocks)
    ctx = _build_context(drafts, full_transcript, style, avoid)

    # Deterministic pass first. It is complete on its own and is the fallback.
    for index, draft in enumerate(drafts):
        rules = _analyse_text(
            draft.text_en,
            _window_counts(drafts, index),
            ctx.global_counts,
            ctx.domain,
            ctx.proper,
        )
        _fill_from_rules(draft, rules, ctx)

    model_used = "rule-enhanced-semantic-planner"
    if _llm_enabled() and drafts:
        payload = _llm_payload(drafts)
        outcome: Optional[_LLMOutcome] = None
        try:
            outcome = await _run_llm_with_budget(payload, full_transcript, style, avoid, cfg)
        except Exception as exc:
            log.warning("planner: Gemini pipeline failed, using rule-based planning: %s", exc)
            warnings.append(f"Gemini 2.5 Pro 2-stage planner fallback triggered: {str(exc)[:120]}")

        if outcome is not None and outcome.brief is not None:
            ctx.brief = outcome.brief
            ctx.palette = _clean_text(outcome.brief.color_palette)
            for draft in drafts:
                _apply_llm(draft, outcome.stage1.get(draft.index), outcome.stage2.get(draft.index), ctx)
            model_used = f"{cfg['model']}-two-stage-cot-cinematic"
            if outcome.errors:
                model_used += "-partial"
                warnings.extend(f"LLM planner: {message}" for message in outcome.errors)

    _apply_diversity(drafts)
    _enforce_cinematography(drafts, ctx)

    planned = [_draft_to_scene(draft) for draft in drafts]
    for draft in drafts:
        warnings.extend(_draft_warnings(draft))
    return planned, warnings, model_used


# =====================================================================
# 14. WORKER HELPERS: CHOOSING THE BEST STOCK CLIP
# =====================================================================

def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_text(candidate: Dict[str, Any]) -> str:
    """Searchable text from provider metadata: tags, titles, and the URL slug."""
    parts: List[str] = []
    for key in ("tags", "title", "description", "alt", "keywords"):
        value = candidate.get(key)
        if isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    parts.append(re.sub(r"https?://[^/]+", " ", str(candidate.get("url") or "")))
    tokens = [_stem(token) for token in _words(" ".join(parts))]
    return f" {' '.join(tokens)} " if tokens else ""


def _stem(token: str) -> str:
    """Light plural stemming, so 'developers' and 'developer' compare equal."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _contains_term(padded_text: str, term: str) -> bool:
    tokens = [_stem(token) for token in _words(term)]
    if not tokens or not padded_text:
        return False
    return f" {' '.join(tokens)} " in padded_text


def _term_hit(padded_text: str, term: str) -> bool:
    """True if the term, or a close synonym of it, appears in the padded text."""
    if _contains_term(padded_text, term):
        return True
    key = " ".join(_stem(token) for token in _words(term))
    return any(_contains_term(padded_text, synonym) for synonym in TERM_SYNONYMS.get(key, ()))


def _query_anchors(query: str) -> List[str]:
    """Content words of a stock query. A matching clip should name at least one."""
    return _dedupe(
        token for token in _words(query)
        if token not in STOPWORDS and token not in GENERIC_QUERY_WORDS and len(token) >= 3
    )


def _subject_check(text: str, scene: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check a clip against the subject words of every query in the shot's ladder.

    A clip is on-subject if it names enough content words of one rung: one word
    for a short rung, two for a longer one. The whole ladder is used because the
    worker does not say which query a clip came from.

    Returns (checkable, hits). checkable is False when no rung has a subject word,
    for example a shot described only by generic words, so nothing can be checked.
    hits are the words of the best-matching rung that the clip names, or [] if no
    rung is matched.
    """
    rungs = [str(rung) for rung in (scene.get("query_ladder") or []) if rung]
    if not rungs:
        rungs = [str(scene.get("visual_query") or "")]
    checkable = False
    best: List[str] = []
    for rung in rungs:
        anchors = _query_anchors(rung)
        if not anchors:
            continue
        checkable = True
        hits = [anchor for anchor in anchors if _term_hit(text, anchor)]
        needed = 1 if len(anchors) <= 2 else 2
        if len(hits) >= needed and len(hits) > len(best):
            best = hits
    return checkable, best


def score_candidate(
    scene: Dict[str, Any],
    candidate: Dict[str, Any],
    rank: int = 0,
    used_ids: Iterable[Any] = (),
) -> Tuple[float, List[str]]:
    """Score one stock candidate for one planned shot. Higher is better.

    Clips that name none of the shot's subject words are rejected (-inf).
    """
    text = _candidate_text(candidate)
    for term in scene.get("negative_terms") or []:
        if _contains_term(text, term):
            return -math.inf, [f"rejected: contains '{term}'"]

    reasons: List[str] = []
    if not text:
        score = 1.0
        reasons.append("no metadata; neutral score")
    else:
        score = 0.0
        for term in str(scene.get("visual_query") or "").split():
            if _contains_term(text, term):
                score += 1.5
        checkable, hits = _subject_check(text, scene)
        if checkable and not hits:
            return -math.inf, ["rejected: no subject of this shot is named in the clip"]
        if hits:
            score += 2.5 * len(hits)
            reasons.append("matches subject: " + ", ".join(hits))

    needed = _as_float(scene.get("duration"))
    length = _as_float(candidate.get("duration"))
    if needed > 0 and length > 0:
        if length >= needed:
            score += 2.0
        elif length < 0.6 * needed:
            score -= 3.0
            reasons.append("clip shorter than 60% of the shot")

    width = int(_as_float(candidate.get("width")))
    height = int(_as_float(candidate.get("height")))
    short_side = min(width, height) if width and height else 0
    if short_side >= 1080:
        score += 1.0
    elif 0 < short_side < 720:
        score -= 2.0
        reasons.append("low-resolution source; will look soft at full HD")
    if width and height:
        # Judge orientation against the output being rendered, not a fixed landscape preference.
        wants_portrait = str(scene.get("aspect_ratio", "16:9")) == "9:16"
        if (height > width) == wants_portrait:
            score += 1.0
        else:
            score -= 0.5
            reasons.append("orientation differs from the output; the renderer will letterbox it")

    identifier = str(candidate.get("id") or candidate.get("url") or "")
    if identifier and identifier in {str(item) for item in used_ids}:
        score -= 5.0
        reasons.append("already used in this video")

    score += max(0.0, 1.0 - 0.1 * rank)
    return score, reasons


def rank_candidates(
    scene: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    used_ids: Iterable[Any] = (),
) -> List[Tuple[float, Dict[str, Any], List[str]]]:
    """Return (score, candidate, reasons) tuples, best first. Rejected items score -inf."""
    used = {str(item) for item in used_ids}
    ranked: List[Tuple[float, Dict[str, Any], List[str]]] = []
    for rank, candidate in enumerate(candidates or []):
        score, reasons = score_candidate(scene, candidate, rank, used)
        ranked.append((score, candidate, reasons))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def choose_best_candidate(
    scene: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    used_ids: Iterable[Any] = (),
    min_score: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """Best acceptable candidate that has not been used yet, or None.

    None means: move to the next query in the ladder. Reuse a clip only as a
    last resort, by passing used candidates in yourself.
    """
    used = {str(item) for item in used_ids}
    for score, candidate, _ in rank_candidates(scene, candidates, used):
        if score < min_score:
            break
        identifier = str(candidate.get("id") or candidate.get("url") or "")
        if identifier and identifier in used:
            continue
        return candidate
    return None


def iter_queries(scene: Dict[str, Any]) -> List[str]:
    """Query ladder for a planned shot, most specific first, broad fallbacks last."""
    ladder = scene.get("query_ladder") or []
    if ladder:
        return list(ladder)
    return _dedupe([
        str(scene.get("visual_query") or ""),
        str(scene.get("secondary_query") or ""),
        str(scene.get("fallback_query") or ""),
    ])
