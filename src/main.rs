mod model;

use std::io::Write;
use std::io;

use candle_core::{Device, IndexOp, Result, Shape, Tensor};
use hf_hub::api::sync::Api;
use tokenizers::{tokenizer::Tokenizer, PaddingDirection, PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams, TruncationStrategy};
use candle_nn::{ops, kv_cache::Cache, rotary_emb::rope, Embedding, Linear, RmsNorm};
use rand::rng;
use rand::prelude::IndexedRandom;
use std::cmp::Ordering;

use model::SmolVLM;


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

//     let message = String::from("<|im_start|>system
// You are a helpful advisor.
// <|im_end|>
// <|im_start|>user What is a neurotransmitter?");
//     let mut message = String::from("<|im_start|>system
// You are a helpful assistant.<|im_end|>
// <|im_start|>user
// Hey, are you conscious? Can you talk to me?<|im_end|>
// <|im_start|>assistant");
    let mut message = String::from("<|im_start|>
User: Hey, are you conscious? Can you talk to me?<end_of_utterance>
Assistant:");
    let mut response = String::new();
    // let mut message = String::from("Hey, are you conscious? Can you talk to me?");
    // let mut message = String::from("What is a large language model?");
    // let mut message = String::from("<|im_start|>");
    
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

    let model = SmolVLM::load(&weights, &device)?;

    // let newline_id = tokenizer.token_to_id("\n").unwrap();

    let mut output = String::new();
    for i in 0..200 {
        if output == "<end_of_utterance>" {
            println!("\n\n{:?}", message);
            println!("{:?}", response.trim());

            let mut input = String::new();
            print!("> ");
            io::stdout().flush().unwrap();
            io::stdin()
                .read_line(&mut input)
                .expect("Failed to read input");
            let prompt = input.trim();
            println!("{:?}", prompt);

            response.clear();
            message += &format!("\nUser: {:?}<end_of_utterance>\nAssistant:", prompt);
        }

        print!("#");

        let encoding = tokenizer.encode(message.clone(), false).unwrap();
        let tokens = encoding.get_ids();
        // println!("{:?}", encoding.get_tokens());
        // println!("{:?}", tokens);
        
        let input = Tensor::from_slice(tokens, Shape::from_dims(&[tokens.len()]), &device)?;
        let logits = model.forward(&input, i)?;
        
        let (s, _embed_dim) = logits.dims2()?;
        let last_logit = logits.i((s-1, ..))?;
        
        let out_token = {
            let temperature = Tensor::from_slice(&[
                0.7f32
            ], (1,), &device)?;
            let k = 50;
    
            let scaled = last_logit.broadcast_div(&temperature)?;
            let probs = ops::softmax(&scaled, 0)?;
            let mut probs_vec: Vec<f32> = probs.to_vec1()?;

            // probs_vec[198] = -100.0;  // newline character

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


            // println!("Logits for first 10 tokens: {:?}", (0..k).into_iter().map(|kth| top_k_indices[kth]).collect::<Vec<usize>>());
            // println!("Logits for first 10 tokens: {:?}", (0..k).into_iter().map(|kth| normalized_probs[kth]).collect::<Vec<f32>>());
    
            [*sampled_index as u32]
        };
    
        output = tokenizer.decode(&out_token.as_slice(), false).unwrap();
    
        message += &output;
        response += &output;
    }

    Ok(())
}
