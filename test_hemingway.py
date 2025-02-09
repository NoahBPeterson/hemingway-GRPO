#!/opt/homebrew/bin/python3.12 -m pytest

import pytest
from hemingway import (
    weak_phrases,
    adverbs_list,
    get_delimiter,
    split_text,
    calculate_reading_level,
    get_readability_level,
    analyze_sentence,
    analyze_paragraph,
    analyze_text
)

# Test data
SAMPLE_TEXT = """The quick brown fox jumps over the lazy dog. 
This is a test paragraph that is actually quite simple.

This is another paragraph. It contains multiple sentences! Does it work well?"""

COMPLEX_TEXT = """In accordance with the aforementioned provisions, I believe it is incumbent upon us to utilize a more sophisticated methodology even if it leaves some shaken. Actually, the implementation of these strategically formulated initiatives, which were previously undertaken by the organizational entities in question, has been demonstrated to facilitate a substantial amelioration in operational efficiency. I think we should expedite the process of evaluating and monitoring the systematically integrated framework. Due to the fact that we have witnessed numerous instances of suboptimal performance, it is essential that we commence a comprehensive assessment of our procedural mechanisms immediately.

Furthermore, in light of the fact that our current operational paradigm has not actually achieved the anticipated level of success, I would suggest that we should strategically realign our organizational infrastructure. The aforementioned analysis, which was meticulously conducted over a substantial time period, clearly indicates that we must proactively address these challenges. Nevertheless, it is particularly important to remember that the successful implementation of these modifications will undoubtedly necessitate a significant allocation of resources."""

COMPLEX_TEXT_2 = """# PROLOGUE: THE SURVIVAL EQUATION

They say no castaway truly understands time until they've watched a sun rise sideways. 

My first dawn on New Lemuria lasted fourteen seconds. 

The second took three days to arrive. 

By the third sunrise—a sickly green flash that set the ammonia clouds screaming—I'd stopped pleading with God and started bargaining with orbital mechanics. My shuttle's broken nav computer spat out fresh three-body predictions every hour, each more despairing than the last. The numbers didn't care that I was Dr. Elara Voss, lead xenobiologist of the Centauri VII expedition. The equations remained merciless as the boiling tides chewing through my landing site. 

I survived the climate swings through stolen alien tricks. 

The Trisolarans left their lessons in the flesh of this dying world—protein chains that hardened into shelters when solar winds screamed, lichens that excreted liquid mathematics, ribcage-like structures that hummed with forgotten orbital harmonics. Their greatest library wasn't in the crystal vaults I'd later crack open, but written in the DNA of the creatures crawling through razor-edged grass. 

But the real revelation came when I blundered into the Gravity Weave. 

No human should've recognized that collapsed dome as artificial. Its support struts were grown from calcified music, its arches curved like frozen desperation. The air inside tasted of ancient sweat and neutron star matter. That's where I found their last monument—a living equation carved from spacetime itself, throbbing with the same arrhythmic pulse as the three suns overhead. 

It offered knowledge. 

It demanded skin. 

When the knowledge rash first bloomed across my palms that night, I thought the black veins were some alien infection. Not until the stars began *speaking*—their gravitational dances translated through itching flesh—did I understand this was the real first contact. The Trisolarans didn't build their library for minds like ours. They built it for survivors. 

This is the story of how I learned to read the universe's scars. 

And why you should fear what's written in mine."""
SAMPLE_SETTINGS = {
    "reading_level_target": "NORMAL"
}

def test_get_delimiter():
    """Test delimiter patterns for different types."""
    assert get_delimiter("paragraph") == r"\n\n+"
    assert get_delimiter("sentence") == r"[.!?]+[\s\n]*"
    assert get_delimiter("word") == r"\s+"
    assert get_delimiter("unknown") == r"\s+"

def test_split_text():
    """Test text splitting functionality."""
    # Test paragraph splitting
    paragraphs = split_text(SAMPLE_TEXT, "paragraph")
    assert len(paragraphs) == 2
    
    # Test sentence splitting
    sentences = split_text("Hello there! How are you? I am fine.", "sentence")
    assert len(sentences) == 3
    
    # Test handling of text without proper punctuation
    text_without_period = "This is a sentence without a period"
    result = split_text(text_without_period, "sentence")
    assert result[0].endswith(".")

def test_calculate_reading_level():
    """Test reading level calculation."""
    stats = {
        "letters": 100,
        "words": 20,
        "sentences": 2
    }
    level = calculate_reading_level(stats)
    assert isinstance(level, int)
    assert level >= 0
    
    # Test edge case with zero words/sentences
    zero_stats = {
        "letters": 0,
        "words": 0,
        "sentences": 0
    }
    assert calculate_reading_level(zero_stats) == 0

def test_get_readability_level():
    """Test readability level determination."""
    settings = {"reading_level_target": "NORMAL"}
    
    # Test normal case
    assert get_readability_level(9, settings, 20) == "normal"
    assert get_readability_level(12, settings, 20) == "hard"
    assert get_readability_level(15, settings, 20) == "very_hard"
    
    # Test too few words
    assert get_readability_level(15, settings, 5) == "normal"
    
    # Test different targets
    technical_settings = {"reading_level_target": "TECHNICAL"}
    assert get_readability_level(15, technical_settings, 20) == "hard"

def test_analyze_sentence():
    """Test sentence analysis."""
    sentence = "I actually believe this sentence is quite complicated."
    stats = analyze_sentence(sentence, SAMPLE_SETTINGS)
    
    print("\nDebug test_analyze_sentence:")
    print(f"Sentence: {sentence}")
    print(f"Stats: {stats}")
    print(f"Highlights: {stats['highlights']}")
    print(f"Found qualifiers: {[phrase for phrase in weak_phrases if phrase in sentence.lower()]}")
    
    assert isinstance(stats, dict)
    assert "characters" in stats
    assert "words" in stats
    assert "highlights" in stats
    assert stats["sentences"] == 1
    assert stats["highlights"]["adverbs"] > 0  # "actually" is in our adverbs list
    assert stats["highlights"]["qualifiers"] > 0  # "I believe" is in our weak phrases

def test_analyze_paragraph():
    """Test paragraph analysis."""
    paragraph = "The quick brown fox jumps. The lazy dog sleeps. Actually, this is interesting!"
    stats = analyze_paragraph(paragraph, SAMPLE_SETTINGS)
    
    print("\nDebug test_analyze_paragraph:")
    print(f"Paragraph: {paragraph}")
    print(f"Split sentences: {split_text(paragraph, 'sentence')}")
    print(f"Stats: {stats}")
    
    assert isinstance(stats, dict)
    assert stats["sentences"] == 3
    assert "highlights" in stats
    assert stats["words"] > 0
    assert stats["letters"] > 0

def test_analyze_text():
    """Test full text analysis."""
    result = analyze_text(SAMPLE_TEXT, SAMPLE_SETTINGS)
    
    assert isinstance(result, dict)
    assert "stats" in result
    assert "paragraphs" in result
    assert "text" in result
    
    stats = result["stats"]
    assert "reading_level" in stats
    assert "readability" in stats
    assert "reading_time_in_secs" in stats
    assert len(result["paragraphs"]) == 2

def test_edge_cases():
    """Test various edge cases."""
    # Empty text
    empty_result = analyze_text("", SAMPLE_SETTINGS)
    assert empty_result["stats"]["words"] == 0
    
    # Single character
    char_result = analyze_text("a", SAMPLE_SETTINGS)
    assert char_result["stats"]["words"] == 1
    
    # Only punctuation
    punct_result = analyze_text("..!?", SAMPLE_SETTINGS)
    assert punct_result["stats"]["words"] == 0
    
    # Multiple consecutive newlines
    newlines_text = "First paragraph.\n\n\n\nSecond paragraph."
    newlines_result = analyze_text(newlines_text, SAMPLE_SETTINGS)
    assert len(newlines_result["paragraphs"]) == 2

def test_specific_features():
    """Test specific features like adverbs, weak phrases, and passive voice."""
    # Test adverbs
    adverb_text = "He actually ran quickly."
    adverb_result = analyze_text(adverb_text, SAMPLE_SETTINGS)
    assert adverb_result["stats"]["highlights"]["adverbs"] > 0
    
    # Test weak phrases
    weak_text = "I believe this is important. I think we should proceed."
    weak_result = analyze_text(weak_text, SAMPLE_SETTINGS)
    assert weak_result["stats"]["highlights"]["qualifiers"] > 0
    
    # Test passive voice
    passive_text = "The ball was thrown by John."
    passive_result = analyze_text(passive_text, SAMPLE_SETTINGS)
    assert passive_result["stats"]["highlights"]["passive_voices"] > 0

def test_complex_text_analysis():
    """Test analysis of a complex, wordy text with multiple features to check."""
    result = analyze_text(COMPLEX_TEXT, SAMPLE_SETTINGS)
    stats = result["stats"]
    
    print("\nDebug test_complex_text_analysis:")
    print(f"Found qualifiers: {[phrase for phrase in weak_phrases if phrase in COMPLEX_TEXT.lower()]}")
    print(f"Highlights: {stats['highlights']}")
    
    # Test basic metrics
    assert stats["words"] > 150  # Should be a long text
    assert stats["sentences"] > 5  # Should have multiple sentences
    assert stats["paragraphs"] == 2  # Should be two paragraphs
    
    # Test readability metrics
    assert stats["reading_level"] > 12  # Should be a high reading level
    assert stats["readability"] in ["hard", "very_hard"]  # Should be difficult to read
    
    # Test specific features
    highlights = stats["highlights"]
    assert highlights["adverbs"] >= 2  # Should catch "actually" and others
    assert highlights["qualifiers"] >= 3  # Should catch "I believe", "I think", etc.
    assert highlights["passive_voices"] >= 2  # Should catch passive constructions
    
    # Test reading time
    assert stats["reading_time_in_secs"] > 30  # Should take significant time to read

if __name__ == "__main__":
    pytest.main([__file__]) 