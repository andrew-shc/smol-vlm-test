mod text_model;
mod vision_model;

use std::io::Write;
use std::io;

use candle_core::{Device, IndexOp, Result, Shape, Tensor};
use hf_hub::api::sync::Api;
use tokenizers::{tokenizer::Tokenizer, PaddingDirection, PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams, TruncationStrategy};
use candle_nn::{ops, kv_cache::Cache, rotary_emb::rope, Embedding, Linear, RmsNorm};
use rand::rng;
use rand::prelude::IndexedRandom;
use std::cmp::Ordering;
use terminal_size::{terminal_size, Width};



fn count_lines(text: &str) -> usize {
    if let Some((Width(w), _)) = terminal_size() {
        // Calculate the number of lines by dividing the text length by the terminal width
        (text.len() + w as usize + 1) / w as usize // Ceiling division
    } else {
        1 // Default to 1 if terminal size is not available
    }
}

fn clear_lines(n: usize) {
    for _ in 0..n {
        print!("\x1B[1A\x1B[2K"); // Move up and clear line
    }
    io::stdout().flush().unwrap();
}


fn main() -> Result<()> {
    /* CHAT TEMPLATE
    <|im_start|>
    {% for message in messages %}
        {{message['role'] | capitalize}}
        {% if message['content'][0]['type'] == 'image' %}
            {{':'}}
        {% else %}
            {{': '}}
        {% endif %}
        
        {% for line in message['content'] %}
            {% if line['type'] == 'text' %}
                {{line['text']}}
            {% elif line['type'] == 'image' %}
                {{ '<image>' }}
            {% endif %}
        {% endfor %}
        
        <end_of_utterance>\n
    {% endfor %}
    {% if add_generation_prompt %}
        {{ 'Assistant:' }}
    {% endif %}
    
     */

    // why are people making smaller inference LLM? is it the future?
    let mut message = String::from("<|im_start|>");
    let mut response = String::new();
    
    let device = Device::new_cuda(0)?;

    let mut tokenizer = Tokenizer::from_pretrained("HuggingFaceTB/SmolVLM-Instruct", None).unwrap();
    // tokenizer.with_padding(Some(
    //     PaddingParams { 
    //         strategy: PaddingStrategy::Fixed(1024),
    //         direction: PaddingDirection::Left,
    //         pad_to_multiple_of: None,
    //         pad_id: 2,
    //         pad_token: "<|im_end|>".to_string(),
    //         pad_type_id: 0,
    //     }
    // ));
    // tokenizer.with_truncation(Some(
    //     TruncationParams { 
    //         direction: TruncationDirection::Left, 
    //         max_length: 1024, // 8192
    //         strategy: TruncationStrategy::LongestFirst, 
    //         stride: 0,
    //     }
    // )).unwrap();
    
    let api = Api::new().unwrap();
    let repo = api.model("HuggingFaceTB/SmolVLM-Instruct".to_string());

    let weights = repo.get("model.safetensors").unwrap();
    let weights = candle_core::safetensors::load(weights, &device)?;

    let model = text_model::SmolVLM::load(&weights, &device)?;

    let mut output = String::new();
    let mut lines_printed = 0;
    for i in 0..10_000 {
        if i == 0 || output == "<end_of_utterance>" {
            // println!("\n\n{:?}", message);
            // println!("{:?}", response.trim());

            let mut input = String::new();
            print!("> ");
            io::stdout().flush().unwrap();
            io::stdin()
                .read_line(&mut input)
                .expect("Failed to read input");
            let prompt = input.trim();
            // println!("{:?}", prompt);

            response.clear();
            message += &format!("\nUser: {}<end_of_utterance>\nAssistant:", prompt);
        }

        // print!("#");
        // io::stdout().flush().unwrap();

        let encoding = tokenizer.encode(message.clone(), false).unwrap();
        let tokens = encoding.get_ids();
        
        let input = Tensor::from_slice(tokens, Shape::from_dims(&[tokens.len()]), &device)?;
        let logits = model.forward(&input, i)?;
        
        let (s, _embed_dim) = logits.dims2()?;
        let last_logit = logits.i((s-1, ..))?;
        
        let out_token = {
            let temperature = Tensor::from_slice(&[
                0.2f32
            ], (1,), &device)?;
            let k = 50;
    
            let scaled = last_logit.broadcast_div(&temperature)?;
            let probs = ops::softmax(&scaled, 0)?;
            let mut probs_vec: Vec<f32> = probs.to_vec1()?;
            let mut indices: Vec<usize> = (0..probs_vec.len()).collect(); 
            indices.sort_by(|&i, &j| probs_vec[j].partial_cmp(&probs_vec[i]).unwrap_or(Ordering::Equal));
            let top_k_indices = &indices[..k];
            let top_k_probs: Vec<f32> = top_k_indices.iter().map(|&i| probs_vec[i]).collect();
            let sum_probs: f32 = top_k_probs.iter().sum();
            let normalized_probs: Vec<f32> = top_k_probs.iter().map(|p| p / sum_probs).collect();
            let mut rng = rng();
            let sampled_index = top_k_indices
                .choose_weighted(&mut rng, |&idx| normalized_probs[top_k_indices.iter().position(|&x| x == idx).unwrap()])
                .expect("Sampling failed");

            [*sampled_index as u32]
        };
    
        output = tokenizer.decode(&out_token.as_slice(), false).unwrap();

        // println!("{:?}", output);
        // println!("{:?}", message);
        if !response.is_empty() {
            clear_lines(lines_printed);
        }
        println!("{:?}", response);
        io::stdout().flush().unwrap();
        lines_printed = count_lines(&response);
    
        message += &output;
        if output != "<end_of_utterance>" {
            response += &output;
        }
    }

    Ok(())
}
