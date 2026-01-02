
import logging

logger = logging.getLogger(__name__)

def restore_oov_words_in_translation(source_text, output_tokens, tokenizer):
    """Restore unknown words from source into translation output"""
    try:
        # Get OOV (out-of-vocabulary) words from source text
        source_words = source_text.split()
        oov_words = []
        
        for word in source_words:
            # Check if word tokenizes to [unk] (unknown token)
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 1 and token_ids[0] == tokenizer.unk_token_id:
                oov_words.append(word)
        
        if not oov_words:
            # No OOV words, decode normally
            return tokenizer.decode(output_tokens, skip_special_tokens=True)
        
        # Decode keeping special tokens to locate <unk>
        raw_translation = tokenizer.decode(output_tokens, skip_special_tokens=False)
        
        # Replace <unk> tokens with original OOV words
        result = raw_translation
        unk_token = tokenizer.unk_token
        
        for oov_word in oov_words:
            if unk_token in result:
                result = result.replace(unk_token, oov_word, 1)
        
        # Clean up any remaining <unk> tokens and normalize whitespace
        result = result.replace(unk_token, '')
        result = ' '.join(result.split())
        
        return result.strip()
        
    except Exception as e:
        logger.warning(f"⚠️ Error restoring OOV words: {e}")
        # Fallback to normal decoding if anything goes wrong
        return tokenizer.decode(output_tokens, skip_special_tokens=True)
