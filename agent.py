import os
import json
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv
from tools import (
    AVAILABLE_TOOLS,
    get_customer_orders,
    calculate_refund,
    send_support_email,
    get_all_customers_spending
)

load_dotenv()

# Universal OpenAI-compatible tool schemas
OPENROUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_orders",
            "description": "Retrieves order history, email, and account status for a specific customer ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID (e.g., 'CUST101')."}
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculates the exact refund amount after applying an optional discount percentage penalty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number", "description": "Total cost of the order."},
                    "discount_percent": {"type": "number", "description": "Percentage deduction to apply (0 to 100)."}
                },
                "required": ["order_amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_support_email",
            "description": "Simulates sending an email notification to a customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_email": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string", "description": "Subject line."},
                    "message_body": {"type": "string", "description": "Email body content."}
                },
                "required": ["recipient_email", "subject", "message_body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_customers_spending",
            "description": "Calculates total order spending across all customers in the database.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Fallback sequence of free tier model slugs
OPENROUTER_FALLBACK_MODELS = [
    "openrouter/free",                               # Dynamic Smart Router
    "nvidia/nemotron-3-ultra-550b-a55b:free",        # High-capacity reasoning
    "google/gemma-4-31b-it:free",                     # Google free open model
    "cohere/north-mini-code:free"                    # Optimized agentic coding & tool use
]

class ToolUseAgent:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        
        if not self.gemini_key and not self.openrouter_key:
            raise ValueError("At least one API key must be provided in environment variables.")
        
        if self.gemini_key:
            self.gemini_client = genai.Client(api_key=self.gemini_key)
        
        if self.openrouter_key:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_key
            )

    def run(self, user_prompt: str, max_steps: int = 5):
        """Attempts multi-step execution on Gemini, falling back through OpenRouter models if rate-limited."""
        # Primary Attempt: Gemini
        if self.gemini_key:
            try:
                return self._run_gemini_multistep(user_prompt, max_steps=max_steps)
            except Exception as e:
                print(f"[Warning] Gemini failed ({e}). Switching to OpenRouter Fallback Chain...")

        # Secondary Attempt: OpenRouter Fallback Chain
        if self.openrouter_key:
            return self._run_openrouter_multistep_chain(user_prompt, max_steps=max_steps)

        raise RuntimeError("All providers failed to process the query.")

    def _run_gemini_multistep(self, user_prompt: str, max_steps: int = 5):
        execution_logs = []
        tools = [get_customer_orders, calculate_refund, send_support_email, get_all_customers_spending]
        
        contents = [user_prompt]
        
        for step in range(max_steps):
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=tools,
                    temperature=0.1
                )
            )
            
            # If function calls were made by model
            if response.function_calls:
                tool_responses = []
                for call in response.function_calls:
                    func_name = call.name
                    func_args = call.args
                    
                    execution_logs.append({"step": step + 1, "tool": func_name, "args": func_args})
                    
                    if func_name in AVAILABLE_TOOLS:
                        result = AVAILABLE_TOOLS[func_name](**func_args)
                        tool_responses.append(
                            types.Part.from_function_response(
                                name=func_name,
                                response={"result": result}
                            )
                        )
                
                # Append assistant model response and tool outputs back into context history
                contents.append(response.candidates[0].content)
                contents.append(types.Content(role="user", parts=tool_responses))
            else:
                # No more function calls; final response delivered
                return {
                    "answer": response.text,
                    "tool_calls": execution_logs,
                    "provider": "Gemini 2.5 Flash Lite",
                    "steps": step + 1
                }

        return {
            "answer": response.text,
            "tool_calls": execution_logs,
            "provider": "Gemini 2.5 Flash Lite",
            "steps": max_steps
        }

    def _run_openrouter_multistep_chain(self, user_prompt: str, max_steps: int = 5):
        """Tries multiple fallback models in series with a multi-step execution loop."""
        for model_id in OPENROUTER_FALLBACK_MODELS:
            try:
                execution_logs = []
                messages = [{"role": "user", "content": user_prompt}]
                
                for step in range(max_steps):
                    response = self.openrouter_client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        tools=OPENROUTER_TOOLS,
                        tool_choice="auto"
                    )
                    
                    response_message = response.choices[0].message
                    tool_calls = response_message.tool_calls
                    
                    if not tool_calls:
                        return {
                            "answer": response_message.content,
                            "tool_calls": execution_logs,
                            "provider": f"OpenRouter ({model_id})",
                            "steps": step + 1
                        }
                    
                    messages.append(response_message)
                    
                    for tool_call in tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        
                        execution_logs.append({"step": step + 1, "tool": func_name, "args": func_args})
                        
                        if func_name in AVAILABLE_TOOLS:
                            result = AVAILABLE_TOOLS[func_name](**func_args)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": func_name,
                                "content": str(result)
                            })
                            
                return {
                    "answer": messages[-1].content,
                    "tool_calls": execution_logs,
                    "provider": f"OpenRouter ({model_id})",
                    "steps": max_steps
                }
            except Exception as e:
                print(f"[Warning] OpenRouter model '{model_id}' failed: {e}. Trying next model...")
                continue

        raise RuntimeError("All models in the OpenRouter Fallback Chain were exhausted.")