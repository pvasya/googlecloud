import { AutoProcessor, AutoTokenizer, Moondream1ForConditionalGeneration, RawImage } from '@huggingface/transformers';

// Load processor, tokenizer and model
const model_id = 'Xenova/moondream2';
const processor = await AutoProcessor.from_pretrained(model_id);
const tokenizer = await AutoTokenizer.from_pretrained(model_id);
const model = await Moondream1ForConditionalGeneration.from_pretrained(model_id, {
    dtype: {
        embed_tokens: 'fp16', // or 'fp32'
        vision_encoder: 'fp16', // or 'q8'
        decoder_model_merged: 'q4', // or 'q4f16' or 'q8'
    }
});

// Prepare text inputs
const prompt = 'Describe.';
const text = `<image>\n\nQuestion: ${prompt}\n\nAnswer:`;
const text_inputs = tokenizer(text);

// Prepare vision inputs
const url = 'https://upload.wikimedia.org/wikipedia/en/9/94/NarutoCoverTankobon1.jpg';
const image = await RawImage.fromURL(url);
const vision_inputs = await processor(image);

// Generate response
const output = await model.generate({
    ...text_inputs,
    ...vision_inputs,
    do_sample: false,
    max_new_tokens: 64,
});
const decoded = tokenizer.batch_decode(output, { skip_special_tokens: false });
console.log(decoded);
// [
//     '<|endoftext|><image>\n\n' +
//     'Question: Describe this image.\n\n' +
//     'Answer: A hand is holding a white book titled "The Little Book of Deep Learning" against a backdrop of a balcony with a railing and a view of a building and trees.<|endoftext|>'
// ]
