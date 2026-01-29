import Prism from 'prismjs';

Prism.languages.fourier = {
    'comment': /\/\/.*|\/\*[\s\S]*?\*\//,
    'string': {
        pattern: /(^|[^\\])"(?:\\.|[^\\"\r\n])*"/,
        lookbehind: true,
        greedy: true
    },
    'keyword': /\b(?:fn|return|if|else|for|while|in|matmul|einsum|gpu|cpu|fast|single|Sequential|Dense|grad|transpose|inverse|dot_product|convolution|pooling|softmax|relu|fft|ifft|sort|quicksort)\b/,
    'constant': /\b(?:c|G|h|k_B|k_e)\b/,
    'boolean': /\b(?:true|false)\b/,
    'function': /\b[a-z_]\w*(?=\s*\()/i,
    'number': /\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[ijk]?\b/,
    'operator': /[-+*/%=<>!&|^~_]+|=>/,
    'punctuation': /[()\[\]{},.;]/
};
