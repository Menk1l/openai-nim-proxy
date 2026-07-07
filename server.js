// server.js - OpenAI to NVIDIA NIM API Proxy with Auto-Retry
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

// 1. MIDDLEWARE: Increased to 50mb to prevent payload errors
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// 2. CONFIGURATION & ENVIRONMENT VARIABLES
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = true;

// 🚀 FAULT TOLERANCE TUNING
const MAX_RETRIES = 4;            // Max attempts to push past NVIDIA's congestion
const INITIAL_RETRY_DELAY = 1500; // Starting pause in milliseconds (1.5 seconds)

// 3. SPEED UPGRADE: Connection Pooling (Ke Keeps connection to NVIDIA open for faster replies)
const axiosInstance = axios.create({
  httpAgent: new http.Agent({ keepAlive: true }),
  httpsAgent: new https.Agent({ keepAlive: true }),
});

const MODEL_MAPPING = {
  'deepseek-flash': 'deepseek-ai/deepseek-v4-flash',
  'deepseek-pro': 'deepseek-ai/deepseek-v4-pro',
  'glm5': 'z-ai/glm5',
  'glm4.7': 'z-ai/glm4.7', 
  'qwen3.5-120': 'qwen/qwen3.5-122b-a10b',
  'qwen3.5-300': 'qwen/qwen3.5-397b-a17b',
  'moonshot': 'moonshotai/kimi-k2.6' 
};

// Helper utility for backoff delays
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'Clean NIM Proxy (No Auth Required)' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(m => ({ id: m, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' }))
  });
});

// MAIN COMPLETIONS ENDPOINT
app.post('/v1/chat/completions', async (req, res) => {
  try {
    let { model, messages, temperature, max_tokens, stream } = req.body;
    
    // 🔥 SCRUB OLD THINK TAGS TO SAVE TOKENS
    let cleanedMessages = (messages || []).map(msg => {
      if (msg.role === 'assistant' && typeof msg.content === 'string') {
        return { ...msg, content: msg.content.replace(/<think>[\s\S]*?<\/think>\s*/g, '') };
      }
      return msg;
    });

    let nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-8b-instruct'; // Default fallback
    
    // Transform request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: cleanedMessages, 
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };
    
    // Custom kwargs attached to root for reasoning models
    if (ENABLE_THINKING_MODE) {
      if (nimModel.includes('glm')) {
        nimRequest.chat_template_kwargs = { 
          enable_thinking: true, 
          clear_thinking: false // MUST be false so GLM outputs native thoughts
        };
      } else {
        nimRequest.chat_template_kwargs = { thinking: true }; 
      }
    }
    
    // --- 🛡️ AUTOMATED RETRY LOOP FOR RESILIENCE ---
    let response;
    let currentDelay = INITIAL_RETRY_DELAY;
    
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        response = await axiosInstance.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          responseType: stream ? 'stream' : 'json',
          timeout: 180000 // 3 minute wait
        });
        break; // If successful, smash out of the loop!
      } catch (error) {
        const errorData = error.response?.data;
        // Parse out any nested error strings from Nvidia
        const errorMsg = typeof errorData === 'string' ? errorData : (errorData?.error?.message || error.message || '');
        const isResourceExhausted = errorMsg.includes('ResourceExhausted') || errorMsg.includes('limit reached');
        const is500Series = error.response?.status >= 500;

        // If it's a server limit issue and we have remaining attempts, back off and try again
        if ((isResourceExhausted || is500Series) && attempt < MAX_RETRIES) {
          console.warn(`⚠️ [Attempt ${attempt}/${MAX_RETRIES}] NVIDIA NIM is throttled (${error.response?.status || 500}). Retrying in ${currentDelay}ms...`);
          await sleep(currentDelay);
          currentDelay *= 2; // Exponential backoff scaling (1.5s -> 3s -> 6s)
        } else {
          throw error; // Hand off to main catch block if it's a non-retryable error or all attempts failed
        }
      }
    }
    // ----------------------------------------------
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ') && !line.includes('[DONE]')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;
                const reasoning = delta.reasoning_content;
                const content = delta.content;
                const isGLM = nimModel.includes('glm');

                if (isGLM) {
                  if (content !== undefined && content !== null) {
                    delta.content = content;
                  } else {
                    delta.content = ""; 
                  }
                  delete delta.reasoning_content;
                  
                } else {
                  if (SHOW_REASONING) {
                    let combinedContent = '';
                    
                    if (reasoning) {
                      if (!reasoningStarted) {
                        combinedContent += '<think>\n';
                        reasoningStarted = true;
                      }
                      combinedContent += reasoning;
                    }
                    
                    if (content !== undefined && content !== null) {
                      if (reasoningStarted && content !== '') {
                        combinedContent += '\n</think>\n\n';
                        reasoningStarted = false;
                      }
                      combinedContent += content;
                    }
                    
                    if (combinedContent !== '' || typeof content === 'string') {
                      delta.content = combinedContent || content || "";
                      delete delta.reasoning_content;
                    }
                  }
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              // Ignore partial JSON parse errors during stream
            }
          } else if (line.includes('[DONE]')) {
            res.write(line + '\n');
          }
        });
      });
      response.data.on('end', () => res.end());
    } else {
      res.json(response.data); 
    }
    
  } catch (error) {
    // 🔥 DETAILED ERROR LOGGING
    console.error('\n=== 🚨 PROXY ERROR CAUGHT 🚨 ===');
    console.error('Status Code:', error.response?.status || 500);
    console.error('Message:', error.message);
    if (error.response?.data) console.error('Details:', JSON.stringify(error.response.data));
    console.error('================================\n');
    
    res.status(error.response?.status || 500).json({
      error: { message: error.response?.data?.detail || error.message || 'Server error' }
    });
  }
});

app.all('*', (req, res) => res.status(404).json({ error: { message: 'Not found' } }));
app.listen(PORT, () => console.log(`🚀 Clean Proxy running on port ${PORT}`));
