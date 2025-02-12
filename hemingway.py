#!/opt/homebrew/bin/python3.12
import re
from pydantic import BaseModel, Field
from typing import Dict, List

# Constants from the original JS file
adverbs_list = {
    "actually": 1,
    "additionally": 1,
    "allegedly": 1,
    "ally": 1,
    "alternatively": 1,
    "anomaly": 1,
    "apply": 1,
    "approximately": 1,
    "ashely": 1,
    "ashly": 1,
    "assembly": 1,
    "awfully": 1,
    "baily": 1,
    "belly": 1,
    "bely": 1,
    "billy": 1,
    "bradly": 1,
    "bristly": 1,
    "bubbly": 1,
    "bully": 1,
    "burly": 1,
    "butterfly": 1,
    "carly": 1,
    "charly": 1,
    "chilly": 1,
    "comely": 1,
    "completely": 1,
    "comply": 1,
    "consequently": 1,
    "costly": 1,
    "courtly": 1,
    "crinkly": 1,
    "crumbly": 1,
    "cuddly": 1,
    "curly": 1,
    "currently": 1,
    "daily": 1,
    "dastardly": 1,
    "deadly": 1,
    "deathly": 1,
    "definitely": 1,
    "dilly": 1,
    "disorderly": 1,
    "doily": 1,
    "dolly": 1,
    "dragonfly": 1,
    "early": 1,
    "elderly": 1,
    "elly": 1,
    "emily": 1,
    "especially": 1,
    "exactly": 1,
    "exclusively": 1,
    "expedite": 1,
    "expend": 1,
    "expiration": 1,
    "facilitate": 1,
    "friendly": 1,
    "frilly": 1,
    "gadfly": 1,
    "gangly": 1,
    "generally": 1,
    "ghastly": 1,
    "giggly": 1,
    "globally": 1,
    "goodly": 1,
    "gravelly": 1,
    "grisly": 1,
    "gully": 1,
    "haily": 1,
    "hally": 1,
    "harly": 1,
    "hardly": 1,
    "heavenly": 1,
    "hillbilly": 1,
    "hilly": 1,
    "holly": 1,
    "holy": 1,
    "homely": 1,
    "homily": 1,
    "horsefly": 1,
    "hourly": 1,
    "immediately": 1,
    "instinctively": 1,
    "imply": 1,
    "italy": 1,
    "jelly": 1,
    "jiggly": 1,
    "jilly": 1,
    "jolly": 1,
    "july": 1,
    "karly": 1,
    "kelly": 1,
    "kindly": 1,
    "lately": 1,
    "likely": 1,
    "lilly": 1,
    "lily": 1,
    "lively": 1,
    "lolly": 1,
    "lonely": 1,
    "lovely": 1,
    "lowly": 1,
    "luckily": 1,
    "mealy": 1,
    "measly": 1,
    "melancholy": 1,
    "mentally": 1,
    "molly": 1,
    "monopoly": 1,
    "monthly": 1,
    "multiply": 1,
    "nightly": 1,
    "oily": 1,
    "only": 1,
    "orderly": 1,
    "panoply": 1,
    "particularly": 1,
    "partly": 1,
    "paully": 1,
    "pearly": 1,
    "pebbly": 1,
    "politically": 1,
    "polly": 1,
    "potbelly": 1,
    "presumably": 1,
    "previously": 1,
    "pualy": 1,
    "quarterly": 1,
    "rally": 1,
    "rarely": 1,
    "recently": 1,
    "rely": 1,
    "reply": 1,
    "reportedly": 1,
    "roughly": 1,
    "sally": 1,
    "scaly": 1,
    "shapely": 1,
    "shelly": 1,
    "shirly": 1,
    "shortly": 1,
    "sickly": 1,
    "silly": 1,
    "sly": 1,
    "smelly": 1,
    "sparkly": 1,
    "spindly": 1,
    "spritely": 1,
    "squiggly": 1,
    "stately": 1,
    "steely": 1,
    "supply": 1,
    "surly": 1,
    "tally": 1,
    "timely": 1,
    "trolly": 1,
    "ugly": 1,
    "underbelly": 1,
    "unfortunately": 1,
    "unholy": 1,
    "unlikely": 1,
    "usually": 1,
    "waverly": 1,
    "weekly": 1,
    "wholly": 1,
    "willy": 1,
    "wily": 1,
    "wobbly": 1,
    "wooly": 1,
    "worldly": 1,
    "wrinkly": 1,
    "yearly": 1
}

weak_phrases = {
    "i believe": 1,
    "i consider": 1,
    "i don't believe": 1,
    "i don't consider": 1,
    "i don't feel": 1,
    "i don't suggest": 1,
    "i don't think": 1,
    "i feel": 1,
    "i hope to": 1,
    "i might": 1,
    "i suggest": 1,
    "i think": 1,
    "i was wondering": 1,
    "i will try": 1,
    "i wonder": 1,
    "in my opinion": 1,
    "is kind of": 1,
    "is sort of": 1,
    "just": 1,
    "maybe": 1,
    "perhaps": 1,
    "possibly": 1,
    "we believe": 1,
    "we consider": 1,
    "we don't believe": 1,
    "we don't consider": 1,
    "we don't feel": 1,
    "we don't suggest": 1,
    "we don't think": 1,
    "we feel": 1,
    "we hope to": 1,
    "we might": 1,
    "we suggest": 1,
    "we think": 1,
    "we were wondering": 1,
    "we will try": 1,
    "we wonder": 1
}

passive_voices = {
    "arisen": "arose",
    "awaken": "awakened",
    "awoken": "awoke",
    "beaten": "beat",
    "been": "be",
    "begun": "began",
    "beheld": "behold",
    "bent": "bent",
    "bidden": "bid",
    "bitten": "bit",
    "bled": "bled",
    "blown": "blew",
    "bought": "bought",
    "broken": "broke",
    "brought": "brought",
    "built": "built",
    "caught": "caught",
    "chosen": "chose",
    "clung": "clung",
    "cut": "cut",
    "dealt": "dealt",
    "done": "did",
    "dove": "dove",
    "drawn": "drew",
    "dreamt": "dreamt",
    "driven": "drove",
    "eaten": "ate",
    "fallen": "fell",
    "fed": "fed",
    "felt": "felt",
    "flown": "flew",
    "forbidden": "forbade",
    "forgiven": "forgave",
    "forgotten": "forgot",
    "forsaken": "forsake",
    "forseen": "foresee",
    "fought": "fought",
    "found": "found",
    "frozen": "froze",
    "given": "gave",
    "gotten": "got",
    "ground": "ground",
    "grown": "grew",
    "hasten": "hasten",
    "heard": "heard",
    "held": "held",
    "hidden": "hid",
    "hit": "hit",
    "hung": "hung",
    "hurt": "hurt",
    "kept": "kept",
    "known": "knew",
    "laid": "laid",
    "led": "led",
    "left": "left",
    "let": "let",
    "lost": "lost",
    "made": "made",
    "meant": "meant",
    "met": "met",
    "outdone": "outdone",
    "outgrown": "outgrown",
    "overseen": "oversee",
    "overtaken": "overtake",
    "overthrown": "overthrow",
    "paid": "paid",
    "proven": "proved",
    "put": "put",
    "read": "read",
    "rewritten": "rewritten",
    "ridden": "rode",
    "risen": "risen",
    "run": "ran",
    "rung": "rang",
    "said": "said",
    "seen": "saw",
    "sent": "sent",
    "sewn": "sewn",
    "shaken": "shook",
    "shaved": "shaved",
    "shone": "shone",
    "shot": "shot",
    "shown": "shown",
    "shrunk": "shrunk",
    "shrunken": "shrunk",
    "shut": "shut",
    "slain": "slew",
    "slid": "slid",
    "sold": "sold",
    "sought": "sought",
    "sown": "sown",
    "spent": "spent",
    "spilt": "spilt",
    "split": "split",
    "spoken": "spoke",
    "spread": "spread",
    "spun": "spun",
    "stolen": "stole",
    "strewn": "strewn",
    "struck": "struck",
    "sung": "sung",
    "sunk": "sunk",
    "sunken": "sunk",
    "swept": "swept",
    "sworn": "swore",
    "swum": "swam",
    "swung": "swung",
    "taken": "took",
    "taught": "taught",
    "thought": "thought",
    "thrown": "threw",
    "told": "told",
    "torn": "tore",
    "undergone": "underwent",
    "understood": "understood",
    "undone": "undone",
    "uprisen": "uprisen",
    "upset": "upset",
    "waken": "waken",
    "withdrawn": "withdrew",
    "woken": "woke",
    "won": "won",
    "worn": "wore",
    "woven": "wove",
    "written": "wrote",
    "wrung": "wrang"
}

too_wordy = {
    "a number of": ["many", "some"],
    "abundance": ["enough", "plenty"],
    "accede to": ["allow", "agree to"],
    "accelerate": ["speed up"],
    "accentuate": ["stress"],
    "accompany": ["go with", "with"],
    "accomplish": ["do"],
    "accorded": ["given"],
    "accrue": ["add", "gain"],
    "acquiesce": ["agree"],
    "acquire": ["get"],
    "additional": ["more", "extra"],
    "adjacent to": ["next to"],
    "adjustment": ["change"],
    "admissible": ["allowed", "accepted"],
    "advantageous": ["helpful"],
    "adversely impact": ["hurt"],
    "advise": ["tell"],
    "aforementioned": ["remove"],
    "aggregate": ["total", "add"],
    "aircraft": ["plane"],
    "all of": ["all"],
    "alleviate": ["ease", "reduce"],
    "allocate": ["divide"],
    "along the lines of": ["like", "as in"],
    "already existing": ["existing"],
    "alternatively": ["or"],
    "ameliorate": ["improve", "help"],
    "anticipate": ["expect"],
    "apparent": ["clear", "plain"],
    "appreciable": ["many"],
    "as a means of": ["to"],
    "as of yet": ["yet"],
    "as to": ["on", "about"],
    "as yet": ["yet"],
    "ascertain": ["find out", "learn"],
    "assistance": ["help"],
    "at this time": ["now"],
    "attain": ["meet"],
    "attributable to": ["because"],
    "authorize": ["allow", "let"],
    "because of the fact that": ["because"],
    "belated": ["late"],
    "benefit from": ["enjoy"],
    "bestow": ["give", "award"],
    "by virtue of": ["by", "under"],
    "cease": ["stop"],
    "close proximity": ["near"],
    "commence": ["begin or start"],
    "comply with": ["follow"],
    "concerning": ["about", "on"],
    "consequently": ["so"],
    "consolidate": ["join", "merge"],
    "constitutes": ["is", "forms", "makes up"],
    "demonstrate": ["prove", "show"],
    "depart": ["leave", "go"],
    "designate": ["choose", "name"],
    "discontinue": ["drop", "stop"],
    "due to the fact that": ["because", "since"],
    "each and every": ["each"],
    "economical": ["cheap"],
    "eliminate": ["cut", "drop", "end"],
    "elucidate": ["explain"],
    "employ": ["use"],
    "endeavor": ["try"],
    "enumerate": ["count"],
    "equitable": ["fair"],
    "equivalent": ["equal"],
    "evaluate": ["test", "check"],
    "evidenced": ["showed"],
    "exclusively": ["only"],
    "expedite": ["hurry"],
    "expend": ["spend"],
    "expiration": ["end"],
    "facilitate": ["ease", "help"],
    "factual evidence": ["facts", "evidence"],
    "feasible": ["workable"],
    "finalize": ["complete", "finish"],
    "first and foremost": ["first"],
    "for the purpose of": ["to"],
    "forfeit": ["lose", "give up"],
    "formulate": ["plan"],
    "honest truth": ["truth"],
    "however": ["but", "yet"],
    "if and when": ["if", "when"],
    "impacted": ["affected", "harmed", "changed"],
    "implement": ["install", "put in place", "tool"],
    "in a timely manner": ["on time"],
    "in accordance with": ["by", "under"],
    "in addition": ["also", "besides", "too"],
    "in all likelihood": ["probably"],
    "in an effort to": ["to"],
    "in between": ["between"],
    "in excess of": ["more than"],
    "in lieu of": ["instead"],
    "in light of the fact that": ["because"],
    "in many cases": ["often"],
    "in order to": ["to"],
    "in regard to": ["about", "concerning", "on"],
    "in some instances ": ["sometimes"],
    "in terms of": ["omit"],
    "in the near future": ["soon"],
    "in the process of": ["omit"],
    "inception": ["start"],
    "incumbent upon": ["must"],
    "indicate": ["say", "state", "or show"],
    "indication": ["sign"],
    "initiate": ["start"],
    "is applicable to": ["applies to"],
    "is authorized to": ["may"],
    "is responsible for": ["handles"],
    "it is essential": ["must", "need to"],
    "literally": ["omit"],
    "magnitude": ["size"],
    "maximum": ["greatest", "largest", "most"],
    "methodology": ["method"],
    "minimize": ["cut"],
    "minimum": ["least", "smallest", "small"],
    "modify": ["change"],
    "monitor": ["check", "watch", "track"],
    "multiple": ["many"],
    "necessitate": ["cause", "need"],
    "nevertheless": ["still", "besides", "even so"],
    "not certain": ["uncertain"],
    "not many": ["few"],
    "not often": ["rarely"],
    "not unless": ["only if"],
    "not unlike": ["similar", "alike"],
    "notwithstanding": ["in spite of", "still"],
    "null and void": ["use either null or void"],
    "numerous": ["many"],
    "objective": ["aim", "fair", "goal"],
    "obligate": ["bind", "compel"],
    "obtain": ["get"],
    "on the contrary": ["but", "so"],
    "on the other hand": ["omit", "but", "so"],
    "one particular": ["one"],
    "optimum": ["best", "greatest", "most"],
    "overall": ["omit"],
    "owing to the fact that": ["because", "since"],
    "participate": ["take part"],
    "particulars": ["details"],
    "pass away": ["die"],
    "pertaining to": ["about", "of", "on"],
    "point in time": ["time", "point", "moment", "now"],
    "portion": ["part"],
    "possess": ["have", "own"],
    "preclude": ["prevent"],
    "previously": ["before"],
    "prior to": ["before"],
    "prioritize": ["rank", "focus on"],
    "procure": ["buy", "get"],
    "proficiency": ["skill"],
    "provided that": ["if"],
    "purchase": ["buy", "sale"],
    "put simply": ["omit"],
    "readily apparent": ["clear"],
    "refer back": ["refer"],
    "regarding": ["about", "of", "on"],
    "relocate": ["move"],
    "remainder": ["rest"],
    "remuneration": ["payment"],
    "require": ["must", "need"],
    "requirement": ["need", "rule"],
    "reside": ["live"],
    "residence": ["house"],
    "retain": ["keep"],
    "satisfy": ["meet", "please"],
    "shall": ["must", "will"],
    "should you wish": ["if you want"],
    "similar to": ["like"],
    "solicit": ["ask for", "request"],
    "span across": ["span", "cross"],
    "strategize": ["plan"],
    "subsequent": ["later", "next", "after", "then"],
    "substantial": ["large", "much"],
    "successfully complete": ["complete", "pass"],
    "sufficient": ["enough"],
    "terminate": ["end", "stop"],
    "the month of": ["omit"],
    "therefore": ["thus", "so"],
    "this day and age": ["today"],
    "time period": ["time", "period"],
    "took advantage of": ["preyed on"],
    "transmit": ["send"],
    "transpire": ["happen"],
    "until such time as": ["until"],
    "utilization": ["use"],
    "utilize": ["use"],
    "validate": ["confirm"],
    "various different": ["various", "different"],
    "whether or not": ["whether"],
    "with respect to": ["on", "about"],
    "with the exception of": ["except for"],
    "witnessed": ["saw", "seen"]
}

def get_delimiter(delimiter_type):
    """Return the appropriate delimiter based on type."""
    if delimiter_type == "paragraph":
        return r"\n\n+"
    elif delimiter_type == "sentence":
        return r"[.!?]+[\s\n]*"
    elif delimiter_type == "word":
        return r"\s+"
    return r"\s+"

def split_text(text, delimiter_type):
    """Split text based on delimiter type."""
    delimiter = get_delimiter(delimiter_type)
    substrings = re.split(delimiter, text)
    results = []
    
    for substring in substrings:
        if substring and not re.match(delimiter, substring):
            if not substring.rstrip().endswith((".", "!", "?")):
                substring = substring.rstrip() + "."
            results.append(substring)
    
    return results

def calculate_reading_level(stats):
    """Calculate reading level based on letters, words, and sentences."""
    letters = stats["letters"]
    words = stats["words"]
    sentences = stats["sentences"]
    
    if words == 0 or sentences == 0:
        return 0
    
    return max(round(letters / words * 4.71 + words / sentences * 0.5 - 21.43), 0)

def get_readability_level(reading_level, parser_settings, word_count):
    """Determine readability level based on reading level and settings."""
    levels = {
        "ACCESSIBLE": {
            "too_few_word_count": 8,
            "hard_readability_level": 8,
            "very_hard_readability_level": 12
        },
        "NORMAL": {
            "too_few_word_count": 14,
            "hard_readability_level": 10,
            "very_hard_readability_level": 14
        },
        "TECHNICAL": {
            "too_few_word_count": 14,
            "hard_readability_level": 14,
            "very_hard_readability_level": 18
        }
    }
    
    level_settings = levels.get(parser_settings.get("reading_level_target", "NORMAL"), levels["NORMAL"])
    
    if word_count < level_settings["too_few_word_count"]:
        return "normal"
    if reading_level >= level_settings["hard_readability_level"] and reading_level < level_settings["very_hard_readability_level"]:
        return "hard"
    if reading_level >= level_settings["very_hard_readability_level"]:
        return "very_hard"
    return "normal"

def analyze_sentence(sentence, settings):
    """Analyze a sentence for various metrics."""
    words = re.findall(r'\b\w+\b', sentence.lower())
    letters = sum(len(word) for word in words)
    
    # Check for weak phrases - using word boundaries for proper matching
    found_qualifiers = []
    sentence_lower = sentence.lower()
    
    # First try exact matches
    for phrase in weak_phrases:
        if re.search(r'\b' + re.escape(phrase) + r'\b', sentence_lower):
            found_qualifiers.append(phrase)
    
    # Then try variations with additional words in between
    # For example: "I would suggest" should match "I would strongly suggest"
    for phrase in weak_phrases:
        parts = phrase.split()
        if len(parts) > 1:
            pattern = r'\b' + r'\b\s+\w+\s+\b'.join(map(re.escape, parts)) + r'\b'
            if re.search(pattern, sentence_lower) and phrase not in found_qualifiers:
                found_qualifiers.append(phrase)
    
    # Check for adverbs
    found_adverbs = []
    for word in words:
        if word in adverbs_list:
            found_adverbs.append(word)
    
    # Check for passive voice
    found_passives = []
    for word in words:
        if word in passive_voices:
            found_passives.append(word)
    
    stats = {
        "characters": len(sentence),
        "letters": letters,
        "words": len(words),
        "sentences": 1,  # Add this back for test compatibility
        "highlights": {
            "adverbs": len(found_adverbs),
            "complex_words": 0,  # Would need implementation
            "grammar_issues": 0,  # Would need implementation
            "hard_sentences": 0,
            "passive_voices": len(found_passives),
            "qualifiers": len(found_qualifiers),
            "very_hard_sentences": 0
        }
    }
    
    return stats

def analyze_paragraph(paragraph, settings):
    """Analyze a paragraph by analyzing its sentences."""
    sentences = split_text(paragraph, "sentence")
    
    stats = {
        "characters": 0,
        "letters": 0,
        "words": 0,
        "sentences": len(sentences),  # Set sentence count once based on actual splits
        "highlights": {
            "adverbs": 0,
            "complex_words": 0,
            "grammar_issues": 0,
            "hard_sentences": 0,
            "passive_voices": 0,
            "qualifiers": 0,
            "very_hard_sentences": 0
        }
    }
    
    for sentence in sentences:
        sentence_stats = analyze_sentence(sentence, settings)
        for key in stats:
            if key == "highlights":
                for highlight_key in stats["highlights"]:
                    stats["highlights"][highlight_key] += sentence_stats["highlights"][highlight_key]
            elif key != "sentences":  # Don't add sentence counts from individual sentences
                stats[key] += sentence_stats.get(key, 0)
    
    return stats

class Highlights(BaseModel):
    adverbs: int = Field(description="Count of adverbs found in the text")
    complex_words: int = Field(description="Count of complex words")
    grammar_issues: int = Field(description="Count of grammar issues detected")
    hard_sentences: int = Field(description="Count of sentences marked as hard to read")
    passive_voices: int = Field(description="Count of passive voice constructions")
    qualifiers: int = Field(description="Count of qualifying/weak phrases")
    very_hard_sentences: int = Field(description="Count of sentences marked as very hard to read")

    model_config = {
        "populate_by_name": True
    }

class TextStats(BaseModel):
    characters: int = Field(description="Total number of characters")
    letters: int = Field(description="Total number of letters")
    words: int = Field(description="Total number of words")
    sentences: int = Field(description="Total number of sentences")
    paragraphs: int = Field(description="Total number of paragraphs")
    highlights: Highlights = Field(description="Analysis highlights of writing issues")
    reading_level: int = Field(description="Calculated reading level score")
    readability: str = Field(description="Readability assessment (normal, hard, or very_hard)")
    reading_time_in_secs: float = Field(description="Estimated reading time in seconds")

    model_config = {
        "populate_by_name": True
    }

class TextAnalysis(BaseModel):
    stats: TextStats = Field(description="Overall statistics of the analyzed text")
    paragraphs: List[str] = Field(description="List of paragraphs in the text")
    text: str = Field(description="The original input text")

    model_config = {
        "populate_by_name": True
    }

    def __getitem__(self, key):
        return self.model_dump()[key]

def analyze_text(text: str, parser_settings: Dict[str, str]) -> TextAnalysis:
    """
    Analyze text for readability and writing style metrics.
    
    Args:
        text: The text to analyze
        parser_settings: Dictionary of parser settings including reading_level_target
        
    Returns:
        TextAnalysis: Complete analysis of the text including statistics and parsed content
    """
    paragraphs = split_text(text, "paragraph")
    all_stats = []
    
    for i, paragraph in enumerate(paragraphs):
        stats = analyze_paragraph(paragraph, parser_settings)
        all_stats.append(stats)
    
    # Calculate overall stats
    overall_stats = {
        "characters": sum(s["characters"] for s in all_stats),
        "letters": sum(s["letters"] for s in all_stats),
        "words": sum(s["words"] for s in all_stats),
        "sentences": sum(s["sentences"] for s in all_stats),
        "paragraphs": len(paragraphs),
        "highlights": {
            "adverbs": sum(s["highlights"]["adverbs"] for s in all_stats),
            "complex_words": sum(s["highlights"]["complex_words"] for s in all_stats),
            "grammar_issues": sum(s["highlights"]["grammar_issues"] for s in all_stats),
            "hard_sentences": sum(s["highlights"]["hard_sentences"] for s in all_stats),
            "passive_voices": sum(s["highlights"]["passive_voices"] for s in all_stats),
            "qualifiers": sum(s["highlights"]["qualifiers"] for s in all_stats),
            "very_hard_sentences": sum(s["highlights"]["very_hard_sentences"] for s in all_stats)
        }
    }
    
    # Calculate additional metrics
    overall_stats["reading_level"] = calculate_reading_level(overall_stats)
    overall_stats["readability"] = get_readability_level(
        overall_stats["reading_level"],
        parser_settings,
        overall_stats["words"]
    )
    overall_stats["reading_time_in_secs"] = overall_stats["words"] / 250 * 60
    
    return TextAnalysis(
        stats=TextStats(
            characters=overall_stats["characters"],
            letters=overall_stats["letters"],
            words=overall_stats["words"],
            sentences=overall_stats["sentences"],
            paragraphs=overall_stats["paragraphs"],
            highlights=Highlights(
                adverbs=overall_stats["highlights"]["adverbs"],
                complex_words=overall_stats["highlights"]["complex_words"],
                grammar_issues=overall_stats["highlights"]["grammar_issues"],
                hard_sentences=overall_stats["highlights"]["hard_sentences"],
                passive_voices=overall_stats["highlights"]["passive_voices"],
                qualifiers=overall_stats["highlights"]["qualifiers"],
                very_hard_sentences=overall_stats["highlights"]["very_hard_sentences"]
            ),
            reading_level=overall_stats["reading_level"],
            readability=overall_stats["readability"],
            reading_time_in_secs=overall_stats["reading_time_in_secs"]
        ),
        paragraphs=paragraphs,
        text=text
    ) 