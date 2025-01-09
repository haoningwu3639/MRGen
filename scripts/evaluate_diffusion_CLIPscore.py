import json
import torch
from PIL import Image
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer

def load_model_and_tokenizer():
    # Load the model and pre-process function of BiomedCLIP
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    return model, preprocess, tokenizer

def calculate_image_similarity(model, preprocess, data, device):
    # Image - Image Similarity
    similarities = []
    for item in tqdm(data):
        image1 = preprocess(Image.open(item['image_path'])).to(device).unsqueeze(0)
        image2 = preprocess(Image.open(item['original_path'])).to(device).unsqueeze(0)
        
        with torch.no_grad():
            image_features1 = model.encode_image(image1)
            image_features2 = model.encode_image(image2)
            similarity = torch.nn.functional.cosine_similarity(image_features1, image_features2)
            similarities.append(similarity.item())
    
    mean_similarity = sum(similarities) / len(similarities)
    print(f'Mean Image-Image Similarity: {mean_similarity}')

def calculate_text_image_similarity(model, preprocess, tokenizer, data, modality_attributes, device):
    similarities = []
    for item in tqdm(data):
        modality = item['modality']
        aux_modality = item['aux_modality']
        region = item['region']
        organs = item['organs']
        
        if modality == 'CT':
            modality_key = " ".join(['Abdomen', modality]).strip()
        else:
            modality_key = " ".join([aux_modality, 'Abdomen', modality]).strip()
        
        modality_attributes_value = modality_attributes.get(modality_key, "")
        prompt = [modality_key, modality_attributes_value, region, organs]
        prompt = ", ".join(filter(None, prompt)).strip()
        
        text_tensor = tokenizer([prompt], context_length=256).to(device)
        image = preprocess(Image.open(item['image_path'])).to(device).unsqueeze(0)
        
        with torch.no_grad():
            text_features = model.encode_text(text_tensor)
            image_features = model.encode_image(image)
            similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
            similarities.append(similarity.item())
    
    mean_similarity = sum(similarities) / len(similarities)
    print(f'Mean Image-Text Similarity: {mean_similarity}')


if __name__ == "__main__":
    model, preprocess, tokenizer = load_model_and_tokenizer()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    with open('./radiopaedia/radiopaedia_synthetic_results_stablediffusion.json', 'r') as f: # 0.3151
    # with open('./radiopaedia/radiopaedia_synthetic_results_finetuned_stablediffusion.json', 'r') as f: 0.6698
    # with open('./radiopaedia/radiopaedia_synthetic_results_MRGen_only_modalities.json', 'r') as f: 0.7512
    # with open('./radiopaedia/radiopaedia_synthetic_results_MRGen_ct_mri.json', 'r') as f: 0.8457
        data = json.load(f)

    modality_attibutes_json = './MRI_diffusion/modality_attributes.json'
    with open(modality_attibutes_json, 'r') as f:
        modality_attributes = json.load(f)

    calculate_image_similarity(model, preprocess, data, device)
    calculate_text_image_similarity(model, preprocess, tokenizer, data, modality_attributes, device)