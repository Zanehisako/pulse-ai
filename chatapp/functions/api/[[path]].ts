import { handleEdgeApiRequest } from '../_edgeApi';

interface PagesContext {
  request: Request;
  env: Record<string, string | undefined>;
}

export const onRequest = async (context: PagesContext): Promise<Response> => {
  return handleEdgeApiRequest(context.request, context.env);
};
