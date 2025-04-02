import streamlit as st
from utils import calculate_parameters, generate_feedback, rewrite_with_llm, meets_criteria, setup_ai_model, test_api_connection, get_available_models, get_model_display_name, get_full_model_name
import time
import nltk
import traceback
from datetime import datetime
from functools import lru_cache
import sys
from streamlit.components.v1 import html
# Load NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')  # Additional resource that might be needed

# Define helper functions first
def reset_inputs():
    """Reset all input fields except API configuration"""
    st.session_state.text_input = ""
    st.session_state.reset_clicked = True

def use_as_input():
    """Replace input text with rewritten text"""
    if 'current_rewritten_text' in st.session_state:
        st.session_state.text_input = st.session_state.current_rewritten_text
        # Force the app to rerun with the new input
        st.rerun()

# Initialize ALL session state variables
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'api_key_status' not in st.session_state:
    st.session_state.api_key_status = None
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'reset_clicked' not in st.session_state:
    st.session_state.reset_clicked = False
if 'text_updated' not in st.session_state:
    st.session_state.text_updated = False

# app.py
st.set_page_config(
    page_title="TextTransmuter Pro",
    page_icon="✨",
    layout="wide"
)

# CSS for both light/dark mode compatibility
st.markdown("""
    <style>
        .title { 
            color: #2E4053;
            font-size: 2.8em !important;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            color: #5D6D7E;
            font-size: 1.2em;
            margin-bottom: 1.5rem;
        }
        .feature-card {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .feature-header {
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        .feature-content {
            margin-top: 1rem;
        }
        ul.feature-list {
            padding-left: 1.5rem;
        }
        ul.feature-list li {
            margin-bottom: 0.5rem;
        }
        @media (prefers-color-scheme: dark) {
            .title { color: #D5D8DC; }
            .subtitle { color: #AEB6BF; }
            .feature-card { border-color: rgba(255, 255, 255, 0.1); }
        }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<p class="title">TextTransmuter Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Text Transformation Suite</p>', unsafe_allow_html=True)

# Collapsible feature card using Streamlit's state
if 'features_expanded' not in st.session_state:
    st.session_state.features_expanded = False

# Card header with toggle button
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.markdown('<div class="feature-header">✨ Key Features</div>', unsafe_allow_html=True)
with col2:
    toggle_button = st.button("▼" if not st.session_state.features_expanded else "▲")

if toggle_button:
    st.session_state.features_expanded = not st.session_state.features_expanded
    st.rerun()

# Card content
if st.session_state.features_expanded:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-content">
                <p>An <strong>enterprise-grade text processing</strong> solution that enables:</p>
                <ul class="feature-list">
                    <li><strong>Precision Adjustment</strong> of readability, sentiment, and objectivity levels</li>
                    <li><strong>Style Transformation</strong> across 15+ professional writing styles</li>
                    <li><strong>Real-time Analytics</strong> with iterative refinement capabilities</li>
                    <li><strong>Advanced Blending</strong> of multiple style parameters</li>
                    <li><strong>Cross-platform</strong> compatibility with major LLM providers</li>
                </ul>
                <p style="font-style: italic; color: #5D6D7E;">Ideal for technical writers, legal professionals, and marketing teams requiring precise textual control.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="feature-card"></div>', unsafe_allow_html=True)


# Cache the model fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_cached_models(provider, api_key):
    """Fetch and cache available models for a provider"""
    return get_available_models(provider, api_key)

# Initialize provider information
AI_PROVIDERS = {
    "Google Gemini": {
        "key_name": "GOOGLE_API_KEY",
        "help": "Enter your Google API key to use Gemini models"
    },
    "OpenAI": {
        "key_name": "OPENAI_API_KEY",
        "help": "Enter your OpenAI API key"
    },
    "DeepSeek": {
        "key_name": "DEEPSEEK_API_KEY",
        "help": "Enter your DeepSeek API key to use DeepSeek models"
    }
}

# Sidebar Configuration
st.sidebar.title("🤖 AI Model Configuration")

# Security Notice
st.sidebar.warning("""
⚠️ **Security Notice:**
- API keys are used only for model access
- Keys are not stored or logged
- Keys are cleared when you close the browser
""")

# AI Provider Selection
ai_provider = st.sidebar.selectbox(
    "Select AI Provider",
    options=list(AI_PROVIDERS.keys()),
    help="Choose which AI service to use for text rewriting"
)

if ai_provider == "DeepSeek":
    st.sidebar.info("""
    **DeepSeek API Note:**
    DeepSeek uses the OpenAI API format but with a different base URL.
    Your API key should be from the DeepSeek platform.
    """)

# API Key Input
api_key = st.sidebar.text_input(
    f"Enter {AI_PROVIDERS[ai_provider]['key_name']}",
    type="password",
    help=AI_PROVIDERS[ai_provider]['help']
)

# Check for API key changes
api_key_changed = False
if api_key:
    current_status = f"{ai_provider}:{api_key}"
    if st.session_state.api_key_status != current_status:
        st.session_state.api_key_status = current_status
        api_key_changed = True

# Fetch models only when needed
if api_key and (len(st.session_state.available_models) == 0 or api_key_changed):
    with st.sidebar:
        with st.spinner("Fetching available models..."):
            try:
                models = fetch_cached_models(ai_provider, api_key)
                if models:
                    st.session_state.available_models = models
                    model_display_names = [get_model_display_name(m) for m in models]
                else:
                    st.warning("Could not fetch available models. Please check your API key.")
                    st.session_state.available_models = []
                    model_display_names = ["No models available"]
            except Exception as e:
                st.error(f"Error fetching models: {str(e)}")
                st.session_state.available_models = []
                model_display_names = ["No models available"]
else:
    model_display_names = ([get_model_display_name(m) for m in st.session_state.available_models] 
                          if len(st.session_state.available_models) > 0 
                          else ["No models available"])

# Model Selection
model_name = st.sidebar.selectbox(
    "Select Model",
    options=model_display_names,
    help="Choose the specific model to use",
    disabled=len(st.session_state.available_models) == 0
)

# Clear Configuration Button
if st.sidebar.button("Clear Configuration", help="Clear saved API key and models"):
    # Reset all session state variables
    st.session_state.available_models = []
    st.session_state.api_key_status = None
    st.session_state.ai_model = None
    st.session_state.api_key = None
    st.session_state.model_name = None
    st.rerun()

# Configuration Status
with st.sidebar.expander("Configuration Status", expanded=False):
    st.write("Provider:", ai_provider)
    st.write("Models Loaded:", len(st.session_state.available_models))
    st.write("API Configured:", st.session_state.ai_model is not None)

# Test Connection Button
if st.sidebar.button(
    "Test Connection",
    disabled=not (api_key and model_name != "No models available"),
    help="Test connection with selected model",
    key="sidebar_test_connection_btn"
):
    with st.sidebar:
        with st.spinner("Testing API connection..."):
            success, message = test_api_connection(ai_provider, api_key, model_name)
            if success:
                st.success(message)
                st.session_state.ai_model = setup_ai_model(ai_provider, api_key, model_name)
                st.session_state.api_key = api_key
                st.session_state.model_name = model_name
            else:
                st.error(message)

# Sidebar Footer
st.sidebar.markdown("""
---
📝 **Note:**
- Models are cached for 1 hour
- Clear configuration if changing API keys
""")

# API configuration check
api_configured = st.session_state.ai_model is not None

# Main UI structure - MAKE SURE TEXT AREA ONLY APPEARS ONCE
if not api_configured:
    # Show blocking message
    st.warning("⚠️ **API Connection Required**\n\nPlease configure your API key and press ((test the connection)) in the sidebar before using the application.")
    

  

# Main columns
col1, col2 = st.columns(2)

with col1:
    st.header("📄 Original Text & Analysis")
    
    # Replace the demo text button and text area with this code
    DEMO_TEXT = """Recent advances in quantum computing have demonstrated the potential for solving complex optimization problems that are intractable for classical computers. The quantum advantage stems from the exploitation of quantum mechanical phenomena such as superposition and entanglement. These properties allow quantum bits (qubits) to exist in multiple states simultaneously, enabling parallel computation at an unprecedented scale."""

    # Create a session state variable for the text input
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""

    # Demo text button
    if st.button("📋 Load Demo Text", help="Load a sample scientific text for testing"):
        st.session_state.text_input = DEMO_TEXT

    # Text input area that uses the session state
    original_text = st.text_area(
        "Paste your text here:",
        value=st.session_state.text_input,
        height=250,
        help="Enter the text you want to analyze and rewrite",
        key="text_area_input"
    )

    # Update session state when text changes, unless the text was just updated by the button
    if original_text != st.session_state.text_input and not st.session_state.text_updated:
        st.session_state.text_input = original_text

    # Reset the text_updated flag after it's been used
    if st.session_state.text_updated:
        st.session_state.text_updated = False

    if original_text:
        initial_params = calculate_parameters(original_text)
        st.subheader("📊 Current Metrics:")
        
        # Create three columns for the metrics
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric(
                "📚 Readability",
                f"{initial_params['readability']:.1f}",
                help="Flesch-Kincaid Grade Level: Lower numbers mean easier to read"
            )
            st.metric(
                "😊 Sentiment",
                f"{initial_params['polarity']:.2f}",
                help="Text sentiment from -1 (very negative) to +1 (very positive)"
            )
            
        with m2:
            st.metric(
                "🎯 Objectivity",
                f"{initial_params['objectivity']:.2f}",
                help="How objective the text is (0 = very subjective, 1 = very objective)"
            )
            st.metric(
                "📏 Avg. Sentence Length",
                f"{initial_params['avg_sentence_length']:.1f}",
                help="Average number of words per sentence"
            )
            
        with m3:
            st.metric(
                "🔤 Lexical Diversity",
                f"{initial_params['lexical_diversity']:.3f}",
                help="Ratio of unique words to total words (higher = more diverse vocabulary)"
            )
            st.metric(
                "📊 Text Statistics",
                f"{initial_params['word_count']} words",
                help=f"Word count: {initial_params['word_count']}, Character count: {initial_params['char_count']}"
            )
    else:
        # Initialize empty initial_params if no text is provided
        initial_params = calculate_parameters("")

with col2:
    st.header("⚙️ Rewriting Settings")
    
    with st.expander("Rewriting Settings", expanded=True):
        # Initialize target_params dictionary at the start of the settings section
        target_params = {}
        
        # Existing parameter controls
        if st.checkbox("Adjust Readability", key="enable_readability"):
            target_readability = st.slider("Target Readability Score", 0.0, 100.0, 60.0, key="readability_slider")
            target_params['readability'] = target_readability

        if st.checkbox("Adjust Objectivity", key="enable_objectivity"):
            target_objectivity = st.slider("Target Objectivity", 0.0, 1.0, 0.5, key="objectivity_slider")
            target_params['objectivity'] = target_objectivity

        if st.checkbox("Adjust Sentiment", key="enable_sentiment"):
            target_sentiment = st.slider("Target Sentiment (-1 negative, 1 positive)", -1.0, 1.0, 0.0, key="sentiment_slider")
            target_params['sentiment'] = target_sentiment

        if st.checkbox("Adjust Lexical Diversity", key="enable_lexical"):
            target_lexical = st.slider("Target Lexical Diversity", 0.0, 1.0, 0.7, key="lexical_slider")
            target_params['lexical_diversity'] = target_lexical

        if st.checkbox("Adjust Word Count", key="enable_wordcount"):
            target_wordcount = st.number_input("Target Word Count", min_value=1, value=100, key="wordcount_input")
            target_params['word_count'] = target_wordcount

        # Text style selector
        text_styles = [
            "As is (no change)",
            "Monologue",
            "Dialogue",
            "Podcast Narrative",
            "Viral Social Media",
            "Public Speech",
            "Professional",
            "Marketing",
            "Academic",
            "Technical",
            "Creative/Fiction",
            "Journalistic",
            "Conversational",
            "Persuasive",
            "Humorous/Satirical",
            "Descriptive/Travel",
            "Legal/Regulatory"
        ]
        
        selected_style = st.selectbox(
            "Text Style",
            options=text_styles,
            key="style_selector"
        )
        
        # Add the style to target_params if not "As is"
        if selected_style != "As is (no change)":
            target_params['style'] = selected_style
        
        # Store parameters in session state for use across the app
        st.session_state.target_params = target_params

        # Make sure we use a unique key for the rewriting button
        rewrite_button = st.button("Start Rewriting", key="start_rewriting_btn_main")

    # Process settings
    st.subheader("🔄 Process Settings")
    max_attempts = st.number_input(
        "Maximum Rewrite Attempts",
        min_value=1,
        max_value=10,
        value=3,
        help="Maximum number of times to attempt rewriting the text"
    )

    if not st.session_state.ai_model and original_text:
        st.warning("⚠️ Please configure and test your AI model connection in the sidebar first!")

# Make sure there's proper indentation after this if statement
if rewrite_button and original_text and st.session_state.ai_model:
    st.divider()
    st.header("🎉 Results")
    
    try:
        with st.spinner('Processing...'):
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status = st.empty()
            
            # Show processing status
            status.info("Starting rewriting process...")
            
            # Collect target parameters
            target_params = {}
            if st.session_state.target_params:
                target_params = st.session_state.target_params
            
            # Process tracking
            current_text = original_text
            
            for attempt in range(max_attempts):
                # Update progress
                progress = (attempt + 1) / max_attempts
                progress_bar.progress(progress)
                status.info(f"Attempt {attempt + 1} of {max_attempts}")
                
                # Calculate current parameters
                current_params = calculate_parameters(current_text)
                
                # Check if we've met the criteria
                if meets_criteria(current_params, target_params):
                    status.success(f"✅ Success! Criteria met after {attempt + 1} attempts.")
                    break
                
                # Generate feedback and rewrite
                feedback = generate_feedback(current_params, target_params)
                rewritten_text, success = rewrite_with_llm(
                    current_text,
                    feedback,
                    st.session_state.ai_model,
                    ai_provider
                )
                
                if not success:
                    status.error(f"❌ Failed to rewrite text: {rewritten_text}")
                    break
                
                current_text = rewritten_text
            
            # Show results
            st.subheader("Text Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Text:**")
                st.text_area("", value=original_text, height=200, disabled=True, key="final_compare_original")
                
            with col2:
                st.markdown("**Rewritten Text:**")
                st.text_area("", value=current_text, height=200, disabled=True, key="final_compare_rewritten")
            
            # Show metrics comparison
            st.subheader("Metrics Comparison")
            final_params = calculate_parameters(current_text)
            initial_params = calculate_parameters(original_text)
            
            cols = st.columns(3)
            with cols[0]:
                st.metric(
                    "Readability",
                    f"{final_params['readability']:.1f}",
                    f"{final_params['readability'] - initial_params['readability']:.1f}"
                )
            with cols[1]:
                st.metric(
                    "Objectivity",
                    f"{final_params['objectivity']:.2f}",
                    f"{final_params['objectivity'] - initial_params['objectivity']:.2f}"
                )
            with cols[2]:
                st.metric(
                    "Sentiment",
                    f"{final_params['polarity']:.2f}",
                    f"{final_params['polarity'] - initial_params['polarity']:.2f}"
                )
            with cols[0]:
                st.metric(
                    "Word Count",
                    f"{final_params['word_count']}",
                    f"{final_params['word_count'] - initial_params['word_count']}"
                )
            with cols[1]:
                st.metric(
                    "Character Count",
                    f"{final_params['char_count']}",
                    f"{final_params['char_count'] - initial_params['char_count']}"
                )
            with cols[2]:
                st.metric(
                    "Lexical Diversity",
                    f"{final_params['lexical_diversity']:.3f}",
                    f"{final_params['lexical_diversity'] - initial_params['lexical_diversity']:.3f}"
                )
                
            # Store the current rewritten text in session state
            st.session_state.current_rewritten_text = current_text
            
            # Add the button with on_click callback
            if current_text != original_text:
                st.button("↩️ Replace Input Text", 
                          on_click=use_as_input,
                          help="Replace the original text with this rewritten version",
                          key="replace_input_btn")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.divider()
st.warning("""
**⚠️ Disclaimer:**
* Results may vary depending on text complexity and AI model capabilities
* Text analysis metrics are approximate
* Always review AI-generated content before use
""")

# Reset the reset_clicked flag after it's been used
if st.session_state.reset_clicked:
    st.session_state.reset_clicked = False

# If you're using session state, you can store the parameters like this:
st.session_state.target_params = target_params

# When the rewrite button is clicked:
#if st.button("Start Rewriting", key="start_rewriting_btn_main"):
#    # Use the stored parameters
#    rewrite_text(text_input, st.session_state.target_params) 

# In the section where you handle the rewriting action
if rewrite_button:
    if original_text.strip():
        with st.spinner("Rewriting text..."):
            try:
                # Make sure we're using the target_params from session state
                target_params = st.session_state.target_params
                
                # Generate rewriting instruction based on parameters
                instruction = generate_feedback(current_params, target_params)
                instruction_text = " ".join(instruction)
                
                # Log what's being sent to the LLM for debugging
                if st.session_state.get('show_debug', False):
                    st.write("Sending instruction to model:", instruction_text)
                    st.write("Target parameters:", target_params)
                
                # Call the rewrite function with all parameters
                rewritten_text = rewrite_with_llm(
                    original_text, 
                    instruction_text, 
                    st.session_state.ai_model,
                    ai_provider
                )
                
                # Store and display results
                st.session_state.current_rewritten_text = rewritten_text
                st.session_state.current_instruction = instruction_text
                
                # ... rest of your results handling code ...
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error(traceback.format_exc())
    else:
        st.warning("Please enter some text to rewrite") 
