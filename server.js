// server.js - OpenAI to NVIDIA NIM API Proxy with Streaming Keep-Alive
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

// FAULT TOLERANCE TUNING
const MAX_RETRIES = 3;            
const INITIAL_RETRY_DELAY = 1500; 

// 3. SPEED UPGRADE: Connection Pooling
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

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'Clean NIM Proxy (Keep-Alive Maintained)' });
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
    
    let cleanedMessages = (messages || []).map(msg => {
      if (msg.role === 'assistant' && typeof msg.content === 'string') {
        return { ...msg, content: msg.content.replace(/<think>[\s\S]*?<\/think>\s*/g, '') };
      }
      return msg;
    });

    let nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-8b-instruct'; 
    
    const nimRequest = {
      model: nimModel,
      messages: cleanedMessages, 
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };
    
    if (ENABLE_THINKING_MODE) {
      if (nimModel.includes('glm')) {
        nimRequest.chat_template_kwargs = { enable_thinking: true, clear_thinking: false };
      } else {
        nimRequest.chat_template_kwargs = { thinking: true }; 
      }
    }
    
    if (stream) {
      // 🚀 1. IMMEDIATELY tell JanitorAI & Railway we are accepting the stream
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      if (res.flushHeaders) res.flushHeaders(); // Lock in the handshake instantly

      // 🚀 2. Start an automated heartbeat interval (Sends an SSE comment every 15s to bypass all timeouts)
      const heartbeat = setInterval(() => {
        try {
          res.write(': keepalive\n\n');
        } catch (err) {
          clearInterval(heartbeat);
        }
      }, 15000);

      let response;
      let currentDelay = INITIAL_RETRY_DELAY;
      
      try {
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
          try {
            response = await axiosInstance.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
              headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
              responseType: 'stream',
              timeout: 600000 // 🚀 Extended to 10 Minutes to allow deep context compilation
            });
            break; 
          } catch (error) {
            const errorData = error.response?.data;
            const errorMsg = typeof errorData === 'string' ? errorData : (errorData?.error?.message || error.message || '');
            const isRetryable = errorMsg.includes('ResourceExhausted') || error.response?.status >= 500;

            if (isRetryable && attempt < MAX_RETRIES) {
              console.warn(`⚠️ Throttled. Retrying attempt ${attempt}...`);
              await sleep(currentDelay);
              currentDelay *= 2;
            } else {
              throw error;
            }
          }
        }

        // 🚀 3. Stop the heartbeat immediately when NVIDIA finally starts streaming real data
        clearInterval(heartbeat);
        
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
                // Ignore parsing fragments
              }
            } else if (line.includes('[DONE]')) {
              res.write(line + '\n');
            }
          });
        });
        response.data.on('end', () => res.end());

      } catch (streamError) {
        clearInterval(heartbeat);
        console.error('🚨 Deepstream Handshake Failure:', streamError.message);
        // Because headers are already sent, we stream the error format cleanly back to the UI
        res.write(`data: ${JSON.stringify({ error: { message: `NVIDIA Pro processing timed out or failed: ${streamError.message}` } })}\n\n`);
        res.end();
      }

    } else {
      // NON-STREAM FALLBACK (Maintained with an extended timeout)
      const response = await axiosInstance.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
        responseType: 'json',
        timeout: 600000 
      });
      res.json(response.data); 
    }
    
  } catch (error) {
    console.error('\n=== 🚨 GLOBAL PROXY ERROR 🚨 ===');
    console.error('Status:', error.response?.status || 500);
    console.error('Message:', error.message);
    console.error('================================\n');
    
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: { message: error.response?.data?.detail || error.message || 'Server error' }
      });
    }
  }
});

app.all('*', (req, res) => res.status(404).json({ error: { message: 'Not found' } }));
app.listen(PORT, () => console.log(`🚀 Resilient Proxy active on port ${PORT}`));
