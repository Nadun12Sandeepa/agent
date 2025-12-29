"""
Multi-Agent AI System with LangChain 1.2.0 Integration
Requires: 
pip install groq langchain==1.2.0 langchain-huggingface sentence-transformers langchain_community langchain-core chromadb beautifulsoup4 requests twilio bytez pillow
"""

import os
import json
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from bs4 import BeautifulSoup
import requests
from groq import Groq
from bytez import Bytez

# Fixed imports for LangChain 1.2.0
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # ← Fixed import
from twilio.rest import Client


class ConfigManager:
    """Manages API keys and configuration"""
    def __init__(self):
        self.config_file = "agent_config.json"
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "groq_api_key": "",
            "bytez_api_key": "",
            "twilio_sid": "",
            "twilio_token": "",
            "twilio_phone": ""
        }
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key, default=""):
        """Get configuration value"""
        # First check environment variables
        env_value = os.getenv(key.upper(), "")
        if env_value:
            return env_value
        # Then check config file
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set configuration value"""
        self.config[key] = value
        self.save_config()


# ================= AGENT SYSTEM =================
class AgentSystem:
    def __init__(self, config_manager):
        self.config = config_manager
        
        # Initialize API clients
        groq_key = self.config.get("groq_api_key")
        bytez_key = self.config.get("bytez_api_key")
        
        self.groq_client = Groq(api_key=groq_key) if groq_key else None
        self.bytez_client = Bytez(bytez_key) if bytez_key else None

        # Twilio credentials
        self.twilio_sid = self.config.get("twilio_sid")
        self.twilio_token = self.config.get("twilio_token")
        self.twilio_phone = self.config.get("twilio_phone")

        # Vector database setup
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            collection_name="agent_data",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )

        self.conversation_history = []
        self.current_image = None
        self.current_video = None

    # ============== WEB SCRAPING AGENT ==============
    def scrape_web_agent(self, url):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(text)

            documents = [
                Document(
                    page_content=chunk,
                    metadata={"source": url, "timestamp": datetime.now().isoformat(), "chunk_id": i}
                ) for i, chunk in enumerate(chunks)
            ]

            self.vector_store.add_documents(documents)

            return {"status": "success", "message": f"Scraped and stored {len(chunks)} chunks from {url}", "chunks": len(chunks)}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ============== QUERY AGENT ==============
    def query_agent(self, query, k=3):
        try:
            if not self.groq_client:
                return {"status": "error", "message": "Groq API key not configured. Please set it in Settings."}
            
            results = self.vector_store.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in results])
            prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )

            answer = completion.choices[0].message.content

            return {
                "status": "success", 
                "answer": answer,
                "sources": [{"source": doc.metadata.get("source", "unknown"), 
                           "content": doc.page_content[:200]} for doc in results]
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ============== SMS AGENT ==============
    def sms_agent(self, phone_number, message):
        try:
            if not all([self.twilio_sid, self.twilio_token, self.twilio_phone]):
                return {"status": "error", "message": "Twilio credentials not configured. Please set them in Settings."}

            client = Client(self.twilio_sid, self.twilio_token)
            sms = client.messages.create(body=message, from_=self.twilio_phone, to=phone_number)

            return {"status": "success", "message": f"SMS sent to {phone_number}", "sid": sms.sid}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ============== IMAGE GENERATION AGENT ==============
    def image_generation_agent(self, prompt):
        try:
            if not self.bytez_client:
                return {"status": "error", "message": "Bytez API key not configured. Please set it in Settings."}
            
            model = self.bytez_client.model("stabilityai/stable-diffusion-xl-base-1.0")
            result = model.run(prompt)
            
            # Handle different return formats
            if isinstance(result, tuple):
                output, error = result
                if error:
                    return {"status": "error", "message": str(error)}
            else:
                output = result
            
            self.current_image = output
            return {"status": "success", "message": "Image generated successfully", "image_url": str(output)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ============== VIDEO GENERATION AGENT ==============
    def video_generation_agent(self, prompt):
        try:
            if not self.bytez_client:
                return {"status": "error", "message": "Bytez API key not configured. Please set it in Settings."}
            
            model = self.bytez_client.model("cerspense/zeroscope_v2_576w")
            result = model.run(prompt)
            
            # Handle different return formats
            if isinstance(result, tuple):
                output, error = result
                if error:
                    return {"status": "error", "message": str(error)}
            else:
                output = result
            
            self.current_video = output
            return {"status": "success", "message": "Video generated successfully", "video_url": str(output)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ============== TRANSLATION AGENT ==============
    def translation_agent(self, text, target_language="en"):
        try:
            if not self.bytez_client:
                return {"status": "error", "message": "Bytez API key not configured. Please set it in Settings."}
            
            model = self.bytez_client.model("google/madlad400-3b-mt")
            prompt = f"<2{target_language}> {text}"
            result = model.run(prompt)
            
            # Handle different return formats
            if isinstance(result, tuple):
                output, error = result
                if error:
                    return {"status": "error", "message": str(error)}
            else:
                output = result
            
            return {"status": "success", "original": text, "translated": str(output), "target_language": target_language}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ============== SUMMARIZATION AGENT ==============
    def summarization_agent(self, text):
        try:
            if not self.bytez_client:
                return {"status": "error", "message": "Bytez API key not configured. Please set it in Settings."}
            
            model = self.bytez_client.model("facebook/bart-large-cnn")
            result = model.run(text)
            
            # Handle different return formats
            if isinstance(result, tuple):
                output, error = result
                if error:
                    return {"status": "error", "message": str(error)}
            else:
                output = result
            
            return {"status": "success", "original_length": len(text), "summary": str(output), "summary_length": len(str(output))}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ============== WORKFLOW ORCHESTRATOR ==============
    def execute_workflow(self, workflow_steps, initial_input, progress_callback=None):
        """
        Execute a series of agents in sequence
        workflow_steps: list of dicts with 'agent' and 'params'
        initial_input: starting data
        progress_callback: function to call with progress updates
        """
        results = []
        current_data = initial_input
        
        for i, step in enumerate(workflow_steps):
            agent_name = step['agent']
            params = step.get('params', {})
            
            if progress_callback:
                progress_callback(f"Step {i+1}/{len(workflow_steps)}: Running {agent_name}...")
            
            try:
                # Route to appropriate agent
                if agent_name == "web_scraping":
                    result = self.scrape_web_agent(current_data)
                    current_data = result.get('message', '')
                    
                elif agent_name == "query":
                    result = self.query_agent(current_data, k=params.get('k', 3))
                    current_data = result.get('answer', '')
                    
                elif agent_name == "summarize":
                    result = self.summarization_agent(current_data)
                    current_data = result.get('summary', '')
                    
                elif agent_name == "translate":
                    target_lang = params.get('target_language', 'es')
                    result = self.translation_agent(current_data, target_lang)
                    current_data = result.get('translated', '')
                    
                elif agent_name == "sms":
                    phone = params.get('phone_number', '')
                    result = self.sms_agent(phone, current_data)
                    
                elif agent_name == "image_generation":
                    result = self.image_generation_agent(current_data)
                    current_data = result.get('image_url', '')
                    
                elif agent_name == "video_generation":
                    result = self.video_generation_agent(current_data)
                    current_data = result.get('video_url', '')
                    
                elif agent_name == "conversation":
                    result = self.conversation_agent(current_data)
                    current_data = result.get('response', '')
                    
                else:
                    result = {"status": "error", "message": f"Unknown agent: {agent_name}"}
                
                results.append({
                    "step": i+1,
                    "agent": agent_name,
                    "result": result,
                    "output_data": current_data
                })
                
                if result.get('status') == 'error':
                    if progress_callback:
                        progress_callback(f"Error in step {i+1}: {result.get('message')}")
                    break
                    
            except Exception as e:
                error_result = {"status": "error", "message": str(e)}
                results.append({
                    "step": i+1,
                    "agent": agent_name,
                    "result": error_result,
                    "output_data": None
                })
                if progress_callback:
                    progress_callback(f"Error in step {i+1}: {str(e)}")
                break
        
        return {
            "status": "success" if all(r['result'].get('status') == 'success' for r in results) else "partial",
            "steps_completed": len(results),
            "results": results,
            "final_output": current_data
        }

    # ============== PREDEFINED WORKFLOWS ==============
    def get_predefined_workflows(self):
        """Return predefined workflow templates"""
        return {
            "web_to_sms": {
                "name": "Web Scraping → Query → SMS",
                "description": "Scrape a website, answer a question about it, and send via SMS",
                "steps": [
                    {"agent": "web_scraping"},
                    {"agent": "query"},
                    {"agent": "sms", "params": {"phone_number": "+1234567890"}}
                ]
            },
            "web_to_summary_translate": {
                "name": "Web → Summarize → Translate",
                "description": "Scrape website, summarize content, translate to another language",
                "steps": [
                    {"agent": "web_scraping"},
                    {"agent": "query"},
                    {"agent": "summarize"},
                    {"agent": "translate", "params": {"target_language": "es"}}
                ]
            },
            "query_to_image": {
                "name": "Query → Generate Image",
                "description": "Query database, use answer to generate an image",
                "steps": [
                    {"agent": "query"},
                    {"agent": "image_generation"}
                ]
            },
            "prompt_to_video_sms": {
                "name": "Text → Video → SMS Link",
                "description": "Generate video from prompt and send link via SMS",
                "steps": [
                    {"agent": "video_generation"},
                    {"agent": "sms", "params": {"phone_number": "+1234567890"}}
                ]
            },
            "full_pipeline": {
                "name": "Full Multi-Agent Pipeline",
                "description": "Scrape → Query → Summarize → Translate → SMS",
                "steps": [
                    {"agent": "web_scraping"},
                    {"agent": "query"},
                    {"agent": "summarize"},
                    {"agent": "translate", "params": {"target_language": "es"}},
                    {"agent": "sms", "params": {"phone_number": "+1234567890"}}
                ]
            }
        }
    def conversation_agent(self, user_message):
        try:
            if not self.groq_client:
                return {"status": "error", "message": "Groq API key not configured. Please set it in Settings → Configure API Keys"}
            
            self.conversation_history.append({"role": "user", "content": user_message})

            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=self.conversation_history,
                temperature=0.8,
                max_tokens=2048
            )

            assistant_message = completion.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return {"status": "success", "response": assistant_message}

        except Exception as e:
            return {"status": "error", "message": str(e)}


# ================= AGENT GUI =================
class AgentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Agent AI System")
        self.root.geometry("1200x800")

        # Initialize config manager
        self.config_manager = ConfigManager()
        self.agent_system = AgentSystem(self.config_manager)

        # Create menu bar
        self.create_menu()

        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Create tabs
        self.create_web_scraping_tab()
        self.create_query_tab()
        self.create_sms_tab()
        self.create_image_tab()
        self.create_video_tab()
        self.create_translation_tab()
        self.create_conversation_tab()
        self.create_workflow_tab()  # New workflow tab

    def create_menu(self):
        """Create menu bar with settings option"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configure API Keys", command=self.show_settings)
        settings_menu.add_separator()
        settings_menu.add_command(label="Exit", command=self.root.quit)

    def show_settings(self):
        """Show settings dialog for API keys"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("API Configuration")
        settings_window.geometry("600x500")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Create a frame with padding
        main_frame = ttk.Frame(settings_window, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(main_frame, text="Configure API Keys", font=('Arial', 14, 'bold')).pack(pady=(0, 20))
        
        # Instructions
        instructions = """Enter your API keys below. These will be saved locally in agent_config.json.
You can also set them as environment variables."""
        ttk.Label(main_frame, text=instructions, wraplength=550, justify='left').pack(pady=(0, 20))
        
        # API Key fields
        fields = {}
        
        # Groq API Key
        ttk.Label(main_frame, text="Groq API Key:", font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(main_frame, text="Get it from: https://console.groq.com/keys", 
                 font=('Arial', 8), foreground='blue').pack(anchor='w')
        fields['groq'] = ttk.Entry(main_frame, width=70, show='*')
        fields['groq'].pack(pady=(5, 15), fill='x')
        fields['groq'].insert(0, self.config_manager.get("groq_api_key"))
        
        # Bytez API Key
        ttk.Label(main_frame, text="Bytez API Key:", font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(main_frame, text="Get it from: https://bytez.com/", 
                 font=('Arial', 8), foreground='blue').pack(anchor='w')
        fields['bytez'] = ttk.Entry(main_frame, width=70, show='*')
        fields['bytez'].pack(pady=(5, 15), fill='x')
        fields['bytez'].insert(0, self.config_manager.get("bytez_api_key"))
        
        # Twilio SID
        ttk.Label(main_frame, text="Twilio Account SID:", font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(main_frame, text="Get it from: https://console.twilio.com/", 
                 font=('Arial', 8), foreground='blue').pack(anchor='w')
        fields['twilio_sid'] = ttk.Entry(main_frame, width=70, show='*')
        fields['twilio_sid'].pack(pady=(5, 15), fill='x')
        fields['twilio_sid'].insert(0, self.config_manager.get("twilio_sid"))
        
        # Twilio Token
        ttk.Label(main_frame, text="Twilio Auth Token:", font=('Arial', 10, 'bold')).pack(anchor='w')
        fields['twilio_token'] = ttk.Entry(main_frame, width=70, show='*')
        fields['twilio_token'].pack(pady=(5, 15), fill='x')
        fields['twilio_token'].insert(0, self.config_manager.get("twilio_token"))
        
        # Twilio Phone
        ttk.Label(main_frame, text="Twilio Phone Number:", font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(main_frame, text="Format: +1234567890", 
                 font=('Arial', 8), foreground='gray').pack(anchor='w')
        fields['twilio_phone'] = ttk.Entry(main_frame, width=70)
        fields['twilio_phone'].pack(pady=(5, 20), fill='x')
        fields['twilio_phone'].insert(0, self.config_manager.get("twilio_phone"))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x')
        
        def save_settings():
            # Save all fields
            self.config_manager.set("groq_api_key", fields['groq'].get())
            self.config_manager.set("bytez_api_key", fields['bytez'].get())
            self.config_manager.set("twilio_sid", fields['twilio_sid'].get())
            self.config_manager.set("twilio_token", fields['twilio_token'].get())
            self.config_manager.set("twilio_phone", fields['twilio_phone'].get())
            
            # Reinitialize agent system with new keys
            self.agent_system = AgentSystem(self.config_manager)
            
            messagebox.showinfo("Success", "API keys saved successfully!")
            settings_window.destroy()
        
        ttk.Button(button_frame, text="Save", command=save_settings).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side='left', padx=5)

    # ---------------- GUI TABS ----------------
    def create_web_scraping_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Web Scraping")

        ttk.Label(frame, text="Enter URL:", font=('Arial', 12)).pack(pady=10)
        self.url_entry = ttk.Entry(frame, width=80)
        self.url_entry.pack(pady=5)
        self.url_entry.insert(0, "https://en.wikipedia.org/wiki/Artificial_intelligence")

        ttk.Button(frame, text="Scrape & Store", command=self.scrape_web).pack(pady=10)
        self.scrape_output = scrolledtext.ScrolledText(frame, height=20, width=100)
        self.scrape_output.pack(pady=10)

    def create_query_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Query Database")

        ttk.Label(frame, text="Enter your question:", font=('Arial', 12)).pack(pady=10)
        self.query_entry = ttk.Entry(frame, width=80)
        self.query_entry.pack(pady=5)

        ttk.Button(frame, text="Search", command=self.query_database).pack(pady=10)
        self.query_output = scrolledtext.ScrolledText(frame, height=20, width=100)
        self.query_output.pack(pady=10)

    def create_sms_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="SMS Agent")

        ttk.Label(frame, text="Phone Number:", font=('Arial', 12)).pack(pady=10)
        self.phone_entry = ttk.Entry(frame, width=50)
        self.phone_entry.pack(pady=5)
        self.phone_entry.insert(0, "+1234567890")

        ttk.Label(frame, text="Message:", font=('Arial', 12)).pack(pady=10)
        self.sms_message = scrolledtext.ScrolledText(frame, height=10, width=80)
        self.sms_message.pack(pady=5)

        ttk.Button(frame, text="Send SMS", command=self.send_sms).pack(pady=10)
        self.sms_output = scrolledtext.ScrolledText(frame, height=8, width=80)
        self.sms_output.pack(pady=10)

    def create_image_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Image Generation")

        ttk.Label(frame, text="Image Prompt:", font=('Arial', 12)).pack(pady=10)
        self.image_prompt = scrolledtext.ScrolledText(frame, height=5, width=80)
        self.image_prompt.pack(pady=5)
        self.image_prompt.insert('1.0', "A futuristic city with flying cars at sunset, cyberpunk style, highly detailed")

        ttk.Button(frame, text="Generate Image", command=self.generate_image).pack(pady=10)
        self.image_output = scrolledtext.ScrolledText(frame, height=15, width=100)
        self.image_output.pack(pady=10)

    def create_video_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Video Generation")

        ttk.Label(frame, text="Video Prompt:", font=('Arial', 12)).pack(pady=10)
        self.video_prompt = scrolledtext.ScrolledText(frame, height=5, width=80)
        self.video_prompt.pack(pady=5)
        self.video_prompt.insert('1.0', "A cat in a wizard hat walking through a magical forest")

        ttk.Button(frame, text="Generate Video", command=self.generate_video).pack(pady=10)
        self.video_output = scrolledtext.ScrolledText(frame, height=15, width=100)
        self.video_output.pack(pady=10)

    def create_translation_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Translation")

        ttk.Label(frame, text="Text to Translate:", font=('Arial', 12)).pack(pady=10)
        self.translate_input = scrolledtext.ScrolledText(frame, height=8, width=80)
        self.translate_input.pack(pady=5)
        self.translate_input.insert('1.0', "Hello, how are you today?")

        ttk.Label(frame, text="Target Language Code (e.g., es, fr, de):", font=('Arial', 10)).pack(pady=5)
        self.lang_entry = ttk.Entry(frame, width=20)
        self.lang_entry.pack(pady=5)
        self.lang_entry.insert(0, "es")

        ttk.Button(frame, text="Translate", command=self.translate_text).pack(pady=10)
        self.translate_output = scrolledtext.ScrolledText(frame, height=10, width=80)
        self.translate_output.pack(pady=10)

    def create_conversation_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Conversation")

        self.conversation_display = scrolledtext.ScrolledText(frame, height=25, width=100)
        self.conversation_display.pack(pady=10)

        input_frame = ttk.Frame(frame)
        input_frame.pack(fill='x', pady=5)

        self.conversation_input = ttk.Entry(input_frame, width=80)
        self.conversation_input.pack(side='left', padx=5)
        self.conversation_input.bind('<Return>', lambda e: self.send_message())

        ttk.Button(input_frame, text="Send", command=self.send_message).pack(side='left')

    def create_workflow_tab(self):
        """Create workflow orchestration tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔗 Workflow")

        # Title
        title_frame = ttk.Frame(frame)
        title_frame.pack(fill='x', pady=10)
        ttk.Label(title_frame, text="Multi-Agent Workflow Orchestrator", 
                 font=('Arial', 14, 'bold')).pack()
        ttk.Label(title_frame, text="Chain multiple agents together to create powerful workflows", 
                 font=('Arial', 10)).pack()

        # Main content frame
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Left side - Workflow builder
        left_frame = ttk.LabelFrame(main_frame, text="Workflow Builder", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        # Predefined workflows
        ttk.Label(left_frame, text="Predefined Workflows:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        workflow_frame = ttk.Frame(left_frame)
        workflow_frame.pack(fill='x', pady=5)
        
        self.workflow_var = tk.StringVar()
        workflows = self.agent_system.get_predefined_workflows()
        
        for key, workflow in workflows.items():
            ttk.Radiobutton(
                workflow_frame, 
                text=workflow['name'],
                variable=self.workflow_var,
                value=key
            ).pack(anchor='w')
        
        self.workflow_var.set("web_to_sms")

        # Input data
        ttk.Label(left_frame, text="Initial Input:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 5))
        ttk.Label(left_frame, text="(URL, text, or prompt depending on first agent)", 
                 font=('Arial', 8)).pack(anchor='w')
        
        self.workflow_input = scrolledtext.ScrolledText(left_frame, height=4, width=50)
        self.workflow_input.pack(fill='x', pady=5)
        self.workflow_input.insert('1.0', "https://en.wikipedia.org/wiki/Artificial_intelligence")

        # Parameters
        ttk.Label(left_frame, text="Parameters (JSON):", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 5))
        ttk.Label(left_frame, text="Override default params (e.g., phone numbers, languages)", 
                 font=('Arial', 8)).pack(anchor='w')
        
        self.workflow_params = scrolledtext.ScrolledText(left_frame, height=6, width=50)
        self.workflow_params.pack(fill='x', pady=5)
        self.workflow_params.insert('1.0', '''{
    "phone_number": "+1234567890",
    "target_language": "es",
    "k": 3
}''')

        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill='x', pady=10)
        
        ttk.Button(button_frame, text="▶ Run Workflow", 
                  command=self.run_workflow, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(button_frame, text="🔄 Clear Results", 
                  command=self.clear_workflow_results).pack(side='left', padx=5)

        # Right side - Results
        right_frame = ttk.LabelFrame(main_frame, text="Workflow Execution Results", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))

        # Progress
        ttk.Label(right_frame, text="Progress:", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.workflow_progress = ttk.Label(right_frame, text="Ready to execute...", 
                                           font=('Arial', 9), foreground='blue')
        self.workflow_progress.pack(anchor='w', pady=5)

        # Results display
        self.workflow_output = scrolledtext.ScrolledText(right_frame, height=30, width=60)
        self.workflow_output.pack(fill='both', expand=True, pady=5)

    # ---------------- GUI ACTIONS ----------------
    def scrape_web(self):
        url = self.url_entry.get()
        self.scrape_output.delete('1.0', tk.END)
        self.scrape_output.insert('1.0', "Scraping web page...\n")

        def run():
            result = self.agent_system.scrape_web_agent(url)
            self.scrape_output.insert(tk.END, f"\n{json.dumps(result, indent=2)}")

        threading.Thread(target=run, daemon=True).start()


    # ---------------- WORKFLOW ACTIONS ----------------
    def run_workflow(self):
        """Execute the selected workflow"""
        workflow_key = self.workflow_var.get()
        initial_input = self.workflow_input.get('1.0', tk.END).strip()
        
        if not initial_input:
            messagebox.showwarning("Input Required", "Please provide initial input for the workflow")
            return
        
        self.workflow_output.delete('1.0', tk.END)
        self.workflow_progress.config(text="Starting workflow...", foreground='blue')
        
        def progress_callback(message):
            """Update progress in GUI"""
            self.workflow_progress.config(text=message, foreground='blue')
            self.workflow_output.insert(tk.END, f"[PROGRESS] {message}\n")
            self.workflow_output.see(tk.END)
        
        def run():
            # Get workflow template
            workflows = self.agent_system.get_predefined_workflows()
            workflow_template = workflows.get(workflow_key)
            
            if not workflow_template:
                self.workflow_output.insert('1.0', "Error: Workflow not found")
                return
            
            # Parse custom parameters
            try:
                params_text = self.workflow_params.get('1.0', tk.END).strip()
                if params_text:
                    custom_params = json.loads(params_text)
                else:
                    custom_params = {}
            except json.JSONDecodeError as e:
                self.workflow_output.insert('1.0', f"Error parsing parameters: {str(e)}")
                return
            
            # Merge custom parameters into workflow steps
            workflow_steps = []
            for step in workflow_template['steps']:
                step_copy = step.copy()
                if 'params' not in step_copy:
                    step_copy['params'] = {}
                step_copy['params'].update(custom_params)
                workflow_steps.append(step_copy)
            
            # Display workflow info
            self.workflow_output.insert('1.0', f"{'='*60}\n")
            self.workflow_output.insert(tk.END, f"WORKFLOW: {workflow_template['name']}\n")
            self.workflow_output.insert(tk.END, f"Description: {workflow_template['description']}\n")
            self.workflow_output.insert(tk.END, f"{'='*60}\n\n")
            self.workflow_output.insert(tk.END, f"Initial Input: {initial_input}\n\n")
            
            # Execute workflow
            result = self.agent_system.execute_workflow(
                workflow_steps, 
                initial_input, 
                progress_callback
            )
            
            # Display results
            self.workflow_output.insert(tk.END, f"\n{'='*60}\n")
            self.workflow_output.insert(tk.END, f"WORKFLOW RESULTS\n")
            self.workflow_output.insert(tk.END, f"{'='*60}\n\n")
            self.workflow_output.insert(tk.END, f"Status: {result['status'].upper()}\n")
            self.workflow_output.insert(tk.END, f"Steps Completed: {result['steps_completed']}\n\n")
            
            for step_result in result['results']:
                self.workflow_output.insert(tk.END, f"\n--- Step {step_result['step']}: {step_result['agent']} ---\n")
                self.workflow_output.insert(tk.END, f"Status: {step_result['result'].get('status')}\n")
                
                if step_result['result'].get('status') == 'success':
                    self.workflow_output.insert(tk.END, f"Output: {step_result['output_data'][:500]}...\n")
                else:
                    self.workflow_output.insert(tk.END, f"Error: {step_result['result'].get('message')}\n")
            
            self.workflow_output.insert(tk.END, f"\n{'='*60}\n")
            self.workflow_output.insert(tk.END, f"FINAL OUTPUT:\n")
            self.workflow_output.insert(tk.END, f"{'='*60}\n")
            self.workflow_output.insert(tk.END, f"{result['final_output']}\n")
            
            # Update progress
            if result['status'] == 'success':
                self.workflow_progress.config(text="✅ Workflow completed successfully!", foreground='green')
            else:
                self.workflow_progress.config(text="⚠️ Workflow completed with errors", foreground='orange')
            
            self.workflow_output.see(tk.END)
        
        threading.Thread(target=run, daemon=True).start()
    
    def clear_workflow_results(self):
        """Clear workflow results"""
        self.workflow_output.delete('1.0', tk.END)
        self.workflow_progress.config(text="Ready to execute...", foreground='blue')

    def query_database(self):
        query = self.query_entry.get()
        self.query_output.delete('1.0', tk.END)
        self.query_output.insert('1.0', "Searching database...\n")

        def run():
            result = self.agent_system.query_agent(query)
            self.query_output.delete('1.0', tk.END)
            if result['status'] == 'success':
                self.query_output.insert('1.0', f"Answer: {result['answer']}\n\n")
                self.query_output.insert(tk.END, "Sources:\n")
                for source in result.get('sources', []):
                    self.query_output.insert(tk.END, f"\n{json.dumps(source, indent=2)}\n")
            else:
                self.query_output.insert('1.0', f"Error: {result['message']}")

        threading.Thread(target=run, daemon=True).start()

    def send_sms(self):
        phone = self.phone_entry.get()
        message = self.sms_message.get('1.0', tk.END).strip()

        def run():
            result = self.agent_system.sms_agent(phone, message)
            self.sms_output.delete('1.0', tk.END)
            self.sms_output.insert('1.0', json.dumps(result, indent=2))

        threading.Thread(target=run, daemon=True).start()

    def generate_image(self):
        prompt = self.image_prompt.get('1.0', tk.END).strip()
        self.image_output.delete('1.0', tk.END)
        self.image_output.insert('1.0', "Generating image...\n")

        def run():
            result = self.agent_system.image_generation_agent(prompt)
            self.image_output.insert(tk.END, f"\n{json.dumps(result, indent=2)}")

        threading.Thread(target=run, daemon=True).start()

    def generate_video(self):
        prompt = self.video_prompt.get('1.0', tk.END).strip()
        self.video_output.delete('1.0', tk.END)
        self.video_output.insert('1.0', "Generating video...\n")

        def run():
            result = self.agent_system.video_generation_agent(prompt)
            self.video_output.insert(tk.END, f"\n{json.dumps(result, indent=2)}")

        threading.Thread(target=run, daemon=True).start()

    def translate_text(self):
        text = self.translate_input.get('1.0', tk.END).strip()
        target_lang = self.lang_entry.get()

        def run():
            result = self.agent_system.translation_agent(text, target_lang)
            self.translate_output.delete('1.0', tk.END)
            self.translate_output.insert('1.0', json.dumps(result, indent=2))

        threading.Thread(target=run, daemon=True).start()

    def send_message(self):
        message = self.conversation_input.get()
        if not message:
            return

        self.conversation_display.insert(tk.END, f"\nYou: {message}\n")
        self.conversation_input.delete(0, tk.END)

        def run():
            result = self.agent_system.conversation_agent(message)
            if result['status'] == 'success':
                self.conversation_display.insert(tk.END, f"AI: {result['response']}\n")
            else:
                self.conversation_display.insert(tk.END, f"Error: {result['message']}\n")
            self.conversation_display.see(tk.END)

        threading.Thread(target=run, daemon=True).start()


# ================= MAIN =================
if __name__ == "__main__":
    root = tk.Tk()
    app = AgentGUI(root)
    root.mainloop()