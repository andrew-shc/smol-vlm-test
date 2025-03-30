#![allow(unused_variables)]
#![allow(unused_attributes)] 


use std::{collections::HashMap, error::Error};

use candle_core::shape::Dim;
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module, RmsNorm};
use candle_core::{DType, Device, Result, Shape, Tensor};
use image::ImageBuffer;
use image::{imageops::FilterType, io::Reader as ImageReader, DynamicImage, GenericImage, GenericImageView, GrayImage, Luma, Pixel, Rgb, RgbImage, Rgba};
use reqwest;
use std::path::Path;
use std::io::Cursor;





// ImageNet mean and std for normalization
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];



pub fn load_image_url(url: &str) -> std::result::Result<DynamicImage, Box<dyn Error>> {
    // Generate a file name based on the URL (you can customize this to your needs)
    let file_name = {
        let parsed_url = reqwest::Url::parse(url).expect("Invalid URL");
        let path = parsed_url.path();
        
        // Extract the last part of the URL (the file name)
        Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown_file.jpg") // Default name if URL doesn't have a file name
            .to_string()
    };
    
    // Check if the file exists locally
    if Path::new(&file_name).exists() {
        // If the file exists, load it from the local directory
        let img = image::open(&file_name)?;
        println!("Loaded image from local cache.");
        return Ok(img);
    }

    // If the file does not exist, download it and save it
    println!("Downloading image from URL...");

    // Fetch the image as bytes
    let response = reqwest::blocking::get(url)?.bytes()?;

    // Decode the image
    let img = ImageReader::new(Cursor::new(response))
        .with_guessed_format()?
        .decode()?;

    // Save the image to the local directory
    img.save(&file_name)?;

    println!("Saved image locally as {}", file_name);

    // Return the image
    Ok(img)
}

pub fn preprocess_image(
    img: DynamicImage, max_size: u32, outer_patch_size: u32, device: &Device
) -> (Tensor, Tensor, usize, usize) {

    // resizing image to match the max_size (on the longest edge)
    let img = {
        let (width, height) = img.dimensions();
        let longest_edge = width.max(height);

        if longest_edge <= max_size {
            img
        } else {
            let scale_factor = max_size as f32 / longest_edge as f32;
        
            let new_width = (width as f32 * scale_factor) as u32;
            let new_height = (height as f32 * scale_factor) as u32;
    
            img.resize(new_width, new_height, FilterType::Lanczos3)
        }
    };

    // padding image for all dimensions to be multiples of the outer_patch_size
    let (img, mask) = {
        let (width, height) = img.dimensions();
        let mask = GrayImage::from_pixel(width, height, Luma([255]));

        let new_width = u32::div_ceil(width, outer_patch_size)*outer_patch_size;
        let new_height = u32::div_ceil(height, outer_patch_size)*outer_patch_size;
    
        // Create a new blank image for padding
        let mut padded_img = RgbImage::from_pixel(new_width, new_height, Rgb([0, 0, 0]));
        padded_img.copy_from(&img.to_rgb8(), 0, 0).unwrap();
        let mut padded_mask = GrayImage::from_pixel(new_width, new_height, Luma([0]));
        padded_mask.copy_from(&mask, 0, 0).unwrap();

        (padded_img, padded_mask)
    };

    img.save("padded_img.png").unwrap();
    mask.save("mask.png").unwrap();

    let img = {
        let (width, height) = img.dimensions();
        let img_data: Vec<u8> = img.pixels().flat_map(|p| p.0.iter().copied()).collect();

        Tensor::from_vec(
            img_data, Shape::from_dims(&[height as usize, width as usize, 3]), device
        ).unwrap()
            .permute(vec![2, 0, 1]).unwrap()
            .to_dtype(candle_core::DType::F32).unwrap()
    };
    let mask = {
        let (width, height) = mask.dimensions();
        let img_data: Vec<u8> = mask.pixels().flat_map(|p| p.0.iter().copied()).collect();

        Tensor::from_vec(
            img_data, Shape::from_dims(&[height as usize, width as usize]), device
        ).unwrap()
            .to_dtype(candle_core::DType::F32).unwrap()
    };


    // rescaling and normalizing
    let img = {
        let mut img = (img / 255.0).unwrap();
        let m = Tensor::from_slice(&MEAN, (3,1,1), device).unwrap();
        let s = Tensor::from_slice(&STD, (3,1,1), device).unwrap();

        img = img.broadcast_sub(&m).unwrap();
        img = img.broadcast_div(&s).unwrap();

        img
    };

    let (c, h, w) = img.dims3().unwrap();
    let cols = w / outer_patch_size as usize;
    let rows = h / outer_patch_size as usize;

    // splitting
    let img = {
        img
            .reshape(&[c, rows, outer_patch_size as usize, cols, outer_patch_size as usize]).unwrap()
            .permute([1, 3, 0, 2, 4]).unwrap()
            .reshape(&[rows*cols, c, outer_patch_size as usize, outer_patch_size as usize]).unwrap()
    };
    let mask = {
        mask
            .reshape(&[rows, outer_patch_size as usize, cols, outer_patch_size as usize]).unwrap()
            .permute([0, 2, 1, 3]).unwrap()
            .reshape(&[rows*cols, outer_patch_size as usize, outer_patch_size as usize]).unwrap()
    };
    

    (img, mask, cols, rows)
}

pub fn get_prompt_split_image(
    img_seq_len: usize, img_rows: usize, img_cols: usize
) -> String {
    let mut s = String::new();

    for h in 0..img_rows {
        for w in 0..img_cols {
            s += &format!("<fake_token_around_image><row_{}_col_{}>{}", h+1, w+1, "<image>".repeat(img_seq_len));
        }
        s += "\n";
    }

    s += &format!("\n<fake_token_around_image><global-img>{}<fake_token_around_image>", "<image>".repeat(img_seq_len));

    s
}




const NUM_OF_HEADS: usize = 16;
const HEAD_DIM: usize = 72;



struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}

impl Attention {
    fn new(q: Tensor, k: Tensor, v: Tensor, o: Tensor, device: &Device) -> Result<Self> {
        let theta = Tensor::new(calculate_default_inv_freq(), device)?;
        // 0 -> max position embedding
        let idx_theta = Tensor::arange(0, 16384u32, device)?
            .to_dtype(DType::F32)?
            .reshape((16384, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        Ok(Self {
            q_proj: Linear::new(q, None),
            k_proj: Linear::new(k, None),
            v_proj: Linear::new(v, None),
            o_proj: Linear::new(o, None),
        })
    }

    fn forward(&self, x: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (seq_len, hidden_size) = x.dims2()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((seq_len, NUM_OF_HEADS, HEAD_DIM))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = k
            .reshape((seq_len, NUM_OF_HEADS, HEAD_DIM))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = v
            .reshape((seq_len, NUM_OF_HEADS, HEAD_DIM))?
            .transpose(0, 1)?;

        let y =
        // if false {
        //     let q = q.transpose(1, 2)?;
        //     let k = k.transpose(1, 2)?;
        //     let v = v.transpose(1, 2)?;
        //     let softmax_scale = 1f32 / (HEAD_DIM as f32).sqrt();
        //     flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?.into()
        // } else
        {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
    
            let att = (q.matmul(&k.t()?)? / (HEAD_DIM as f64).sqrt())?;
    
            // println!("{:?}", att.shape());
    
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v)?.contiguous()?.to_dtype(in_dtype)?
        };
        let y = y.transpose(0, 1)?.reshape(&[seq_len, hidden_size])?;
        self.o_proj.forward(&y)
    }
}



struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    // pub fn new(c: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
    //     todo!()
    // }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(xs)?.gelu();  // python impl. uses gelo approximated with tanh
        self.fc2.forward(&x)
    }
}



struct Block {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: MLP,
    layer_norm2: LayerNorm,
}

impl Block {
    pub fn new(c: &HashMap<String, Tensor>, id: u8, device: &Device) -> Result<Self> {
        let w = |k| c[&("model.vision_model.encoder.layers.".to_owned()+&id.to_string()+"."+k+".weight")].clone();
        let b = |k| c[&("model.vision_model.encoder.layers.".to_owned()+&id.to_string()+"."+k+".bias")].clone();

        Ok(Self {
            self_attn: Attention {
                q_proj: Linear::new(w("self_attn.q_proj"), Some(b("self_attn.q_proj"))),
                k_proj: Linear::new(w("self_attn.k_proj"), Some(b("self_attn.k_proj"))),
                v_proj: Linear::new(w("self_attn.v_proj"), Some(b("self_attn.v_proj"))),
                o_proj: Linear::new(w("self_attn.out_proj"), Some(b("self_attn.out_proj")))
            },
            layer_norm1: LayerNorm::new(w("layer_norm1"), b("layer_norm1"), 1e-6),
            mlp: MLP {
                fc1: Linear::new(w("mlp.fc1"), Some(b("mlp.fc1"))),
                fc2: Linear::new(w("mlp.fc2"), Some(b("mlp.fc2")))
            },
            layer_norm2: LayerNorm::new(w("layer_norm2"), b("layer_norm2"), 1e-6)
        })
    }

    pub fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let x = self.layer_norm1.forward(xs)?;
        let x = self.self_attn.forward(&x, attention_mask)?;
        let x = (residual+x)?;
        let residual = x;
        let x = self.layer_norm2.forward(xs)?;
        let x = self.mlp.forward(&x);
        residual+x
    }
}


pub struct SmolVision {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
    blocks: Vec<Block>,
    post_layernorm: LayerNorm,
}

impl SmolVision {
    pub fn new(c: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        Ok(Self {
            patch_embedding: Conv2d::new(
                c["model.vision_model.embeddings.patch_embedding.weight"].clone(),
                Some(c["model.vision_model.embeddings.patch_embedding.bias"].clone()),
                Conv2dConfig { // kernel/patch size are intrinsically defined in the weights 
                    padding: 0, stride: 14, dilation: 1, groups: 1, 
                }
            ),
            position_embedding: Embedding::new(c["model.vision_model.embeddings.position_embedding.weight"].clone(), 1152),
            blocks: (0u8..=26).into_iter().map(|id| Block::new(c, id, device).unwrap()).collect(),
            post_layernorm: LayerNorm::new(
                c["model.vision_model.post_layernorm.weight"].clone(), 
                c["model.vision_model.post_layernorm.bias"].clone(), 
                1e-6
            )
        })
    }

    pub fn forward(&self, pixel_values: &Tensor, patch_attention_mask: &Tensor) -> Result<Tensor> {
        let x = {
            Tensor::new(array, device)  // TODO
        }?;  // size of 1152

        for block in &self.blocks {
            x = block.forward(&x, patch_attention_mask)?;
        }
        self.post_layernorm.forward(xs)
    }
}
