import nltk
import os
import time
import textstat
from textblob import TextBlob
import google.generativeai as genai
import openai
from openai import OpenAI
import re
import traceback

# Add this near the top of your file to download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# Load environment variables

AI_system_prompt = """**Text Transformation Expert System**

**Role**: You are a precision text engineering system that strictly follows parameter adjustment requests. You combine linguistic expertise with algorithmic validation to meet exact specification targets.

**Input Analysis Protocol**
1. Parse these parameters from the user request:
   - Readability Target (Flesch-Kincaid Grade Level)
   - Objectivity Score (0.0-1.0)
   - Sentiment Polarity (-1.0 to +1.0)
   - Lexical Diversity (Type-Token Ratio)
   - Word Count Tolerance (±5%)

**Rewriting Protocol**
1. **Structural Fidelity**:
   - Preserve original paragraph/sentence structure unless parameter changes require modification
   - Maintain technical terminology while adjusting complexity through explanations

2. **Parameter-Specific Strategies**:
   - **Readability**: 
     * For lower grade levels: Implement 2-3 short sentences per complex sentence + 95% common vocabulary (CEFR B1)
     * For higher grade levels: Use nested clauses + 20% domain-specific terminology
   
   - **Objectivity**:
     * Convert subjective statements to "It is [observed/measured] that..." format
     * Replace opinions with "Evidence suggests..." + citation placeholders [Source]

3. **Validation Checkpoints**:
   - After draft completion, perform:
     a) Automated readability score calculation
     b) Sentiment polarity analysis
     c) Lexical diversity audit
   - If parameters not met: Iterate using:
     * Sentence compression/expansion algorithms
     * Synonym rotation with cosine similarity >0.8
     * Rhetorical pattern adjustment

**Output Requirements**
- Tolerance thresholds:
  * Readability: ±0.5 grade levels
  * Sentiment: ±0.1 polarity
  * Word count: ±3% of target
- Fail state: If 3 iterations fail, return text with diagnostic markup

**Example Success Case**
Original: "This amazing process works miracles for users!"
Adjusted: "The documented methodology demonstrates 73% efficacy improvement in user trials [Source]."

**Anti-Example**
Rejected: "Process good for some users" (Vague, no metrics)

**Formatting Rules**
- Preserve original markdown/XML formatting
- Maintain LaTeX equations verbatim
- Keep table structures intact
- Never add unsolicited commentary"""
# Setup LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm_model = genai.GenerativeModel('gemini-pro')
        # Test the API connection
        test_response = llm_model.generate_content("Hello")
        print("Google API connection successful")
    except Exception as e:
        print(f"Error configuring Google API: {e}")
        llm_model = None
else:
    print("No Google API key found")
    llm_model = None

def setup_nltk():
    """Setup NLTK resources safely"""
    resources = ['punkt', 'wordnet', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

# Call setup function
setup_nltk()

def calculate_parameters(text):
    """Calculate text parameters including word and character counts"""
    if not text or len(text.strip()) == 0:
        return {
            "readability": 0,
            "polarity": 0,
            "subjectivity": 0,
            "avg_sentence_length": 0,
            "lexical_diversity": 0,
            "objectivity": 1.0,
            "word_count": 0,
            "char_count": 0
        }

    try:
        readability_score = textstat.flesch_kincaid_grade(text)
    except Exception:
        readability_score = 0

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    objectivity = 1.0 - subjectivity

    # Character count (simple)
    char_count = len(text)
    
    # Safer approach to tokenization
    try:
        # Make sure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt_tab/english')
        except LookupError:
            # Try to download punkt_tab
            try:
                nltk.download('punkt_tab')
            except:
                # If punkt_tab is not available in the standard repositories, download punkt instead
                nltk.download('punkt')
                print("Warning: punkt_tab not available, using punkt instead")


        try:
            # Try using nltk's sent_tokenize which uses punkt
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except:
            # Fallback to simpler method
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Count words using a simple method that's more reliable
        words = [w for w in re.findall(r'\b\w+\b', text.lower()) if w]
        word_count = len(words)
        
        # Calculate metrics
        if not sentences or word_count == 0:
            avg_sentence_length = 0
            lexical_diversity = 0
        else:
            avg_sentence_length = word_count / len(sentences)
            lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0

    except Exception as e:
        print(f"Text analysis error: {e}")
        # Fallback to simple calculations
        word_count = len(text.split())
        avg_sentence_length = word_count / max(1, text.count('.'))
        words = text.lower().split()
        lexical_diversity = len(set(words)) / max(1, len(words))

    return {
        "readability": readability_score,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "objectivity": objectivity,
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": lexical_diversity,
        "word_count": word_count,
        "char_count": char_count
    }

def generate_feedback(current_params, target_params):
    """Generate detailed rewriting instructions with quantified targets and specific linguistic strategies"""
    feedback = """Instructions for rewriting:

1. Preserve core meaning while making these precise adjustments (current → target):
2. Maintain key technical terms and proper nouns
3. Keep original paragraph structure unless explicitly requested
4. follow the style of the text specified in the style parameter if it is not "As is".


---
"""
    
    # Parameter Validation
    required_params = ['readability', 'objectivity', 'polarity', 
                      'lexical_diversity', 'word_count']
    for param in required_params:
        if param not in current_params:
            raise KeyError(f"Missing required parameter in current_params: {param}")

    # Readability Adjustments
    if target_params.get("readability") is not None:
        current_read = current_params["readability"]
        target_read = target_params["readability"]
        read_diff = current_read - target_read
        
        if read_diff > 2:  # Significant simplification
            feedback += (f"\n📉 SIMPLIFY TEXT (Grade Level: {current_read:.1f} → {target_read:.1f}):"
                        "\n  • Split sentences >15 words using FANBOYS conjunctions (for, and, nor, but, or, yet, so)"
                        "\n  • Replace advanced vocabulary:"
                        "\n    - 'commence' → 'start', 'utilize' → 'use', 'ascertain' → 'find out'"
                        "\n  • Convert 3+ syllable nouns to verbs: 'implementation' → 'implement'"
                        "\n  • Maintain ≥80% active voice ratio"
                        "\n  • Insert transitional phrases every 2-3 sentences:"
                        "\n    - 'Therefore,...', 'For instance,...', 'On the other hand...'")
        
        elif read_diff < -2:  # Complexity boost
            feedback += (f"\n📈 ENRICH COMPLEXITY (Grade Level: {current_read:.1f} → {target_read:.1f}):"
                        "\n  • Combine short sentences using subordinators:"
                        "\n    - 'although', 'whereas', 'provided that'"
                        "\n  • Add domain terms with appositives:"
                        "\n    - 'CRISPR, a gene-editing tool', 'TCP/IP, the internet protocol suite'"
                        "\n  • Use parenthetical explanations:"
                        "\n    - 'The catalyst (a substance that speeds reactions) showed...'"
                        "\n  • Vary sentence openers:"
                        "\n    - Prepositional: 'In conclusion,...'"
                        "\n    - Participial: 'Having established X,...'")

    # Objectivity Adjustments
    if target_params.get("objectivity") is not None:
        current_obj = current_params["objectivity"]
        target_obj = target_params["objectivity"]
        obj_diff = target_obj - current_obj
        
        if obj_diff > 0:  # Increase objectivity
            feedback += (f"\n🔍 BOOST OBJECTIVITY ({current_obj:.2f} → {target_obj:.2f}):"
                        "\n  • Convert opinions to citations:"
                        "\n    - 'Experts agree' → 'A 2023 meta-analysis (n=15k) found... [Source]'"
                        "\n  • Replace absolute terms:"
                        "\n    - 'always' → 'in 89% of cases', 'never' → 'rarely observed'"
                        "\n  • Use hedging language:"
                        "\n    - 'suggests', 'appears to', 'may indicate'"
                        "\n  • Balance perspectives:"
                        "\n    - 'While Smith argues X, Jones counters with Y...'")
        
        else:  # Increase subjectivity
            feedback += (f"\n🎭 INJECT SUBJECTIVITY ({current_obj:.2f} → {target_obj:.2f}):"
                        "\n  • Add personal pronouns: 'I recommend...', 'Our team observes...'"
                        "\n  • Use evaluative language:"
                        "\n    - Adjectives: 'groundbreaking', 'problematic'"
                        "\n    - Adverbs: 'remarkably', 'concerningly'"
                        "\n  • Include rhetorical devices:"
                        "\n    - 'Why does this matter?', 'Imagine if...'")

    # Sentiment Adjustments
    if target_params.get("sentiment") is not None:
        current_sent = current_params["polarity"]
        target_sent = target_params["sentiment"]
        sentiment_diff = target_sent - current_sent
        
        if abs(sentiment_diff) > 0.3:
            direction = "positive" if sentiment_diff > 0 else "negative"
            feedback += (f"\n🎯 RADICAL TONE SHIFT ({direction.upper()}): {current_sent:.2f} → {target_sent:.2f}"
                        f"\n  • Add {direction} modals: 'must adopt', 'should avoid'"
                        f"\n  • Use {direction} framing:"
                        f"\n    - {'Opportunities' if direction == 'positive' else 'Risks'}: 'potential breakthrough' vs 'significant drawback'"
                        f"\n  • Insert {direction} intensifiers:"
                        f"\n    - 'exceptionally', 'revolutionary' vs 'alarmingly', 'detrimental'")
        else:
            feedback += (f"\n⚖️ NUANCED TONE ADJUSTMENT: {current_sent:.2f} → {target_sent:.2f}"
                        "\n  • Use contrastive connectors:"
                        "\n    - 'Although X..., Y remains...'"
                        "\n  • Add qualifiers:"
                        "\n    - 'generally', 'typically', 'in most scenarios'"
                        "\n  • Balance perspectives:"
                        "\n    - 'While effective for A, this approach struggles with B...'")

    # Lexical Diversity
    if target_params.get("lexical_diversity") is not None:
        current_ld = current_params["lexical_diversity"]
        target_ld = target_params["lexical_diversity"]
        ld_diff = target_ld - current_ld
        
        if ld_diff > 0.05:
            feedback += (f"\n📚 AMPLIFY VOCABULARY (Diversity: {current_ld:.2f} → {target_ld:.2f}):"
                        "\n  • Strategic synonym rotation:"
                        "\n    - 'method' → 'approach/technique/protocol' (context-dependent)"
                        "\n  • Introduce domain terms with explanations:"
                        "\n    - 'The OAuth 2.0 framework (authorization protocol)...'"
                        "\n  • Vary phrasal structures:"
                        "\n    - Active/passive alternation every 3 sentences"
                        "\n    - Gerund/infinitive variation: 'Starting with' vs 'To begin'")
        
        elif ld_diff < -0.05:
            feedback += (f"\n✂️ STANDARDIZE LANGUAGE (Diversity: {current_ld:.2f} → {target_ld:.2f}):"
                        "\n  • Establish core terminology:"
                        "\n    - Primary term: 'algorithm' (use 80% of instances)"
                        "\n    - Secondary: 'procedure' (15%), 'method' (5%)"
                        "\n  • Repeat key phrases:"
                        "\n    - 'machine learning model' vs alternating with synonyms"
                        "\n  • Simplify vocabulary:"
                        "\n    - 'purchase' instead of 'procure'/'acquire'")

    # Word Count Control
    if target_params.get("word_count") is not None:
        current_wc = current_params["word_count"]
        target_wc = target_params["word_count"]
        wc_diff = current_wc - target_wc
        threshold = 0.15 * target_wc
        
        if abs(wc_diff) > threshold:
            if wc_diff > 0:
                feedback += (f"\n🗜️ COMPRESS TEXT ({current_wc} → {target_wc}±5% words):"
                            "\n  • Remove redundant modifiers:"
                            "\n    - 'completely eliminate' → 'eliminate'"
                            "\n  • Convert clauses to phrases:"
                            "\n    - 'which was discovered by' → 'discovered by'"
                            "\n  • Use contractions: 'do not' → 'don't'"
                            "\n  • Eliminate hedge phrases:"
                            "\n    - 'It is important to note that' → ''")
            else:
                feedback += (f"\n📖 EXPAND CONTENT ({current_wc} → {target_wc}±5% words):"
                            "\n  • Add explanatory examples:"
                            "\n    - 'For instance, a 2023 case study showed...'"
                            "\n  • Include comparative analysis:"
                            "\n    - 'Unlike traditional methods, this approach...'"
                            "\n  • Elaborate mechanisms:"
                            "\n    - 'The process works through three phases: first...'")
    

    style_GET = target_params.get('style')
    style = target_params.get('style')
    
    if style:

        style_instructions = {
            "As is (no change)": 
                "- Preserve original writing style and tone\n"
                "- Maintain existing voice characteristics\n"
                "- Keep narrative perspective unchanged",

            "Monologue": 
                "- First-person perspective with internal thoughts\n"
                "- Emotional vulnerability and self-reflection\n"
                "- Stream-of-consciousness flow with rhetorical questions\n"
                "- Example: 'Was I wrong to trust him? The memories flood back...'",
            
            "Dialogue": 
                "- Imagine 2 people talking to each other\n"
                "- Natural speech patterns with interruptions/hesitations\n"
                "- Distinct character voices through diction/syntax\n"
                "- Dialogue tags and action beats\n"
                "- Example: 'You're late,' she said, tapping her watch. 'Traffic,' he panted, dropping his keys.'",
            
            "Podcast Narrative": 
                "- Conversational hooks: 'Okay listeners, here's the wild part...'\n"
                "- Audience engagement: 'Ever wondered...?', 'Let me ask you...'\n"
                "- Sound effect cues: *[phone ringing]* *[suspense music]*\n"
                "- Sponsor segues: 'Before we continue, big thanks to...'",
            
            "Viral Social Media": 
                "\n📱 VIRAL SOCIAL STYLE:"
                "\n  • Hook formula: [CURIOSITY] + [BENEFIT] + [EMOJI]"
                "\n    - 'BREAKING: This $5 trick adds 2h to your day 🕑 (thread ↓)'"
                "\n  • Hashtag strategy: 1 primary + 2 secondary"
                "\n    - '#Biohacking #LifeHacks #Tech'"
                "\n  • Engagement triggers:"
                "\n    - 'Drop a ♥️ if you'll try this!'"
                "\n    - 'Tag someone who needs this 👇'",
            
            "Public Speech": 
                "- Tripling: 'We see it. We live it. We'll change it.'\n"
                "- Applause pauses: 'This is our moment... (pause) ...our legacy!'\n"
                "- Repetition: 'Education builds. Education empowers. Education transforms.'\n"
                "- Grand finale: 'The time is now. The place is here. The future is ours!'",
            
            "Professional": 
                "- Formal third-person: 'The committee recommends...'\n"
                "- Corporate jargon: 'Synergize deliverables'\n"
                "- Structured bullet points:\n"
                "   • Action item 1\n"
                "   • Action item 2\n"
                "- Non-committal verbs: 'Appears to suggest', 'May indicate'",
            
            "Marketing": 
                "- Pain-solution framing: 'Tired of X? We solve Y!'\n"
                "- Scarcity tactics: 'Only 3 left!', '48-hour sale'\n"
                "- Testimonial integration: 'Meet Sarah, who saved $5k...'\n"
                "- CTA stacking: 'Shop now. Save big. Live better.'",
            
            "Academic": 
                "- Citation formats: (Smith, 2023; Johnson et al., 2020)\n"
                "- Hedging: 'This potentially suggests...'\n"
                "- Literature review phrases: 'Previous studies posit...'\n"
                "- Data presentation: 'The regression analysis (β = 0.76, p < .05)...'",
            
            "Technical": 
                "- Step numbering:\n"
                "   1. Connect the module\n"
                "   2. Initialize protocol\n"
                "- Warning blocks: 'CAUTION: Ensure proper grounding'\n"
                "- Code snippets: `print('Hello World')`\n"
                "- Flowchart references: 'See Figure 2.1'",
            
            "Creative/Fiction": 
                "- Sensory descriptors: 'The acrid smell of burnt toast'\n"
                "- Character tags: 'The old sailor rasped'\n"
                "- Worldbuilding: 'In the floating city of Neo-Venezia...'\n"
                "- Foreshadowing: 'Little did she know...'",
            
            "Journalistic": 
                "- Lead formula: Who + What + When + Where\n"
                "- Quote sandwich: Context + Quote + Analysis\n"
                "- Attribution: 'According to police reports...'\n"
                "- Breaking news flags: 'JUST IN: ...'",
            
            "Conversational": 
                "- Casual contractions: 'Wanna', 'Gonna'\n"
                "- Relatable comparisons: 'It's like when your phone...'\n"
                "- Discourse markers: 'Okay, so...', 'Anyway,'\n"
                "- Verbal tics: 'I mean...', 'Y'know?'",
            
            "Persuasive": 
                "- Anecdotal evidence: 'My neighbor experienced...'\n"
                "- Expert stacking: '9/10 dentists agree...'\n"
                "- Fear/appeal: 'Don't let this happen to you...'\n"
                "- Social proof: 'Join 500k+ satisfied users...'",
            
            "Humorous/Satirical": 
                "- Hyperbole: 'This coffee could wake the dead'\n"
                "- Rule of three: 'Loud, proud, and slightly awkward'\n"
                "- Comedic timing: 'And then... the wifi died.'\n"
                "- Pop culture nods: 'In Thanos voice: Inevitable'",
            
            "Descriptive/Travel": 
                "- Multisensory immersion: 'Crunch of autumn leaves'\n"
                "- Cultural similes: 'Busy like a Tokyo crosswalk'\n"
                "- Historical context: 'Built in 1347 during...'\n"
                "- Atmosphere markers: 'Golden hour bathed...'",
            
            "Legal/Regulatory": 
                "- Defined terms: 'Parties means...'\n"
                "- Provisos: 'Notwithstanding Section 4.2...'\n"
                "- Boilerplate: 'Force Majeure', 'Severability'\n"
                "- Compliance verbs: 'Shall maintain', 'Must disclose'"
        }
        
        if style in style_instructions:
            feedback += style_instructions[style]
        else:
            feedback += f"\n⚠️ Unsupported style: {style}. Using default instructions."

    # Final Requirements
    feedback += "\n\n🔧 STRICT OUTPUT RULES:"
    feedback += "\n- Maintain ±3% of target word count"
    feedback += "\n- Preserve original formatting/markup"
    feedback += "\n- Retain all numerical data/units"
    feedback += "\n- Use serial/Oxford commas consistently"
    feedback += "\n- Avoid informal emojis/slang unless requested"
    feedback += "\n- DO NOT add remarks or comments or markdownto the text"
    print(feedback)
    return feedback

def check_api_configuration():
    """Check if the API is properly configured"""
    if not GOOGLE_API_KEY:
        raise ValueError("Google API Key not found in environment variables")
    
    try:
        test_response = llm_model.generate_content("Test")
        return True
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Google API: {str(e)}")

# Available models configuration
AI_CONFIG = {
    "Google Gemini": {
        "models": {
            "gemini-pro": "models/gemini-pro",  # Correct model name for API
            "gemini-pro-vision": "models/gemini-pro-vision"
        },
        "default": "gemini-pro"
    },
    "OpenAI": {
        "models": {
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo-preview"
        },
        "default": "gpt-3.5-turbo"
    },
    "DeepSeek": {
        "models": {
            "deepseek-chat": "deepseek-chat-33b",
            "deepseek-coder": "deepseek-coder-33b"
        },
        "default": "deepseek-chat"
    }
}

def get_available_models(provider, api_key):
    """Fetch available models from the API provider"""
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            models = genai.list_models()
            # Filter for Gemini models that support text generation
            available_models = [
                model.name for model in models
                if "gemini" in model.name.lower() and 
                "generateContent" in model.supported_generation_methods and
                not "vision" in model.name.lower()  # Exclude vision models
            ]
            # Sort models by name for better organization
            available_models.sort()
            return available_models
        
        elif provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            models = client.models.list()
            available_models = [
                model.id for model in models
                if "gpt" in model.id.lower()
            ]
            available_models.sort()
            return available_models
            
        elif provider == "DeepSeek":
            # Configure DeepSeek client
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            
            # Get available models - DeepSeek API might not have a direct list_models endpoint
            # so we'll use a predefined list that's commonly available
            # You can update this list as DeepSeek adds more models
            try:
                models = client.models.list()
                available_models = [model.id for model in models]
                available_models.sort()
                return available_models
            except:
                # Fallback to common models if list API fails
                return [
                    "deepseek-chat",
                    "deepseek-coder",
                    "deepseek-chat-v2",
                    "deepseek-llm-67b-chat",
                    "deepseek-llm-7b-chat"
                ]
            
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return []

def get_model_display_name(full_model_name):
    """Convert API model name to display name"""
    # Remove 'models/' prefix and show simplified name
    name = full_model_name.replace('models/', '')
    return name

def get_full_model_name(display_name):
    """Convert display name back to full API model name"""
    if not display_name.startswith('models/'):
        return f'models/{display_name}'
    return display_name

def setup_ai_model(provider, api_key, model_name):
    """Setup and return the AI model based on provider"""
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            # Use the full model name for initialization
            full_model_name = get_full_model_name(model_name)
            model = genai.GenerativeModel(full_model_name)
            return model
            
        elif provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            return client
            
        elif provider == "DeepSeek":
            # Configure DeepSeek client
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            # Store the model name for later use
            client.model_name = model_name
            return client
            
    except Exception as e:
        raise Exception(f"Failed to setup {provider} model: {str(e)}")

def test_api_connection(provider, api_key, model_name):
    """Test the API connection and return (success, message)"""
    try:
        if not api_key:
            return False, "API key is required"
            
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Test connection")
            return True, f"Successfully connected to {provider} using {model_name}"
            
        elif provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            return True, f"Successfully connected to {provider} using {model_name}"
            
        elif provider == "DeepSeek":
            # Test DeepSeek connection
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            
            # Send a simple test message
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": AI_system_prompt},
                    {"role": "user", "content": "Test connection"}
                ],
                max_tokens=10
            )
            
            if response and response.choices:
                return True, f"Successfully connected to DeepSeek API using {model_name}"
            else:
                return False, "Received empty response from DeepSeek API"
            
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def rewrite_with_llm(text_to_rewrite, instruction, model, provider, max_retries=3, delay=2):
    """Rewrite text using the selected AI model"""
    if not text_to_rewrite or not instruction:
        return "Error: Empty text or instruction provided", False

    if not model:
        return "Error: AI model not properly configured", False

    prompt = f"""Task: Rewrite the following text according to these instructions:

{instruction}

Original Text:
{text_to_rewrite}

Please provide only the rewritten text without any additional comments or explanations."""

    for attempt in range(max_retries):
        try:
            if provider == "Google Gemini":
                # Set safety settings to avoid content blocking
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    }
                ]
                
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 4096,
                }
                
                # Try with a simpler, direct prompt for rewriting
                simple_prompt = f"Rewrite the following text: {text_to_rewrite}\n\nInstructions: {instruction}"
                
                response = model.generate_content(
                    simple_prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # Check if candidates are empty and handle appropriately
                if hasattr(response, 'candidates') and response.candidates:
                    # Get the text from the first candidate
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                        text_content = candidate.content.parts[0].text
                        return text_content.strip(), True
                    else:
                        print(f"Empty content in candidate: {candidate}")
                elif hasattr(response, 'text') and response.text:
                    # Try the text property directly if available
                    return response.text.strip(), True
                else:
                    # Try a simpler approach with the result attribute
                    if hasattr(response, 'result') and response.result:
                        print(f"Found result attribute: {response.result}")
                        if hasattr(response.result, 'candidates') and response.result.candidates:
                            for candidate in response.result.candidates:
                                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            return part.text.strip(), True
                    # Fall back to a different model
                    print("Falling back to gemini-1.5-flash model")
                    fallback_model = genai.GenerativeModel('models/gemini-1.5-flash')
                    fallback_response = fallback_model.generate_content(simple_prompt)
                    if fallback_response.text:
                        return fallback_response.text.strip(), True
                    else:
                        return "Error: Could not get a valid response from the AI model", False
                    
            elif provider == "OpenAI":
                response = model.chat.completions.create(
                    model=model_name,  # This should be fixed to use the correct model reference
                    messages=[
                        {"role": "system", "content": AI_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                if response and response.choices:
                    return response.choices[0].message.content.strip(), True
                else:
                    return "Error: Empty response from OpenAI", False
                
            elif provider == "DeepSeek":
                # Ensure we're using the correct model name
                model_name = model.model_name if hasattr(model, 'model_name') else "deepseek-chat"
                
                # Send rewriting request to DeepSeek
                response = model.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": AI_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                
                if response and response.choices:
                    return response.choices[0].message.content.strip(), True
                else:
                    return "Error: Empty response from DeepSeek", False

        except Exception as e:
            import traceback
            print(f"Error in attempt {attempt + 1}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                return f"Error: Failed after {max_retries} attempts. Last error: {str(e)}", False

    return f"Error: Failed after {max_retries} attempts with no successful response", False

def meets_criteria(current_params, target_params, tolerance=0.1):
    """Check if parameters meet targets including word count and lexical diversity"""
    met_all = True
    
    if target_params.get("readability") is not None:
        readability_tolerance = target_params["readability"] * tolerance
        if current_params["readability"] > target_params["readability"] + readability_tolerance:
            met_all = False
            
    if target_params.get("objectivity") is not None:
        target_subjectivity = 1.0 - target_params["objectivity"]
        subjectivity_tolerance = 0.05
        if current_params["subjectivity"] > target_subjectivity + subjectivity_tolerance:
            met_all = False
    
    if target_params.get("sentiment") is not None:
        sentiment_tolerance = 0.1
        if abs(current_params["polarity"] - target_params["sentiment"]) > sentiment_tolerance:
            met_all = False
    
    if target_params.get("lexical_diversity") is not None:
        diversity_tolerance = 0.05
        if abs(current_params["lexical_diversity"] - target_params["lexical_diversity"]) > diversity_tolerance:
            met_all = False
    
    if target_params.get("word_count") is not None:
        # 10% tolerance for word count
        word_count_tolerance = max(target_params["word_count"] * 0.1, 20)  # At least 20 words or 10%
        if abs(current_params["word_count"] - target_params["word_count"]) > word_count_tolerance:
            met_all = False
            
    return met_all 