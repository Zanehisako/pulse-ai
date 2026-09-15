/**
 * Edge API Configuration for Cloudflare Pages Functions
 * Fully config-driven: defines defaults, timeouts, model IDs, endpoints, and clinical prompts.
 */

export interface EdgeConfigDefinition {
  remote_llm: {
    default_api_url: string;
    default_model: string;
    chat_completions_path: string;
    health_path: string;
    timeout_seconds: number;
    temperature: number;
    max_tokens: number;
    system_prompt: string;
  };
  routes: {
    status: string;
    predict_stream: string;
    health: string;
  };
  cors: {
    allow_origins: string[];
    allow_methods: string;
    allow_headers: string;
  };
}

export const rawConfig: EdgeConfigDefinition = {
  remote_llm: {
    default_api_url: 'https://yassinetakiko--pulseai-vllm-backend-serve.modal.run',
    default_model: 'Qwen/Qwen2.5-7B-Instruct',
    chat_completions_path: '/v1/chat/completions',
    health_path: '/health',
    timeout_seconds: 110.0,
    temperature: 0.1,
    max_tokens: 1024,
    system_prompt:
      'You are the PulseAI Clinical Decision Support Agent for blood donor eligibility and hospital inventory management. When evaluating donor eligibility, reference standard clinical guidelines (age 17-75, weight >= 50kg / 110 lbs, minimum 56 days between whole blood donations). Provide clear, structured reasoning. Conclude with a clear clinical note emphasizing that AI recommendations serve as decision support and do not replace professional medical judgment.',
  },
  routes: {
    status: '/api/ml/orchestrator/status',
    predict_stream: '/api/ml/predict/nl',
    health: '/api/health',
  },
  cors: {
    allow_origins: ['*'],
    allow_methods: 'GET, POST, OPTIONS',
    allow_headers: 'Content-Type, Authorization, Accept, X-Requested-With',
  },
};

export default rawConfig;
