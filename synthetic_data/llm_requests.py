import sys
import os

from openai import OpenAI
from google import genai

import pickle
import asyncio
from tqdm.autonotebook import tqdm, trange
import numpy as np
import random

#TODO: Hard code this or set the path in the environment variable
build_project_path = os.environ['BUILD_PROJECT_PATH']
synthetic_data_path = os.path.join(build_project_path, 'synthetic_data')

if synthetic_data_path not in sys.path:
    sys.path.append(synthetic_data_path)

from gpt_parsing import parse_gpt_response

data_path = os.path.join(build_project_path, 'synthetic_data', 'data')

with open(os.path.join(data_path, 'initial_prompt.txt'), 'r') as f:
    base_prompt = f.read()

example_query_titles = []
with open(os.path.join(data_path, 'example_query_titles.txt'), 'r') as f:
    for line in f:
        example_query_titles.append(line.strip())

with open(os.path.join(data_path, 'example_assistant_response.txt'), 'r') as f:
    example_assistant_response = f.read()

with open(os.path.join(data_path, 'follow_up_prompt.txt'), 'r') as f:
    follow_up_prompt = f.read()

rng = np.random.default_rng()

def get_client(api_key_name="SAMBANOVA_API_KEY", base_url="https://api.sambanova.ai/v1", use_google_api=False):

    if use_google_api:
        return genai.Client(api_key=os.environ['GEMINI_API_KEY'])
     
    return OpenAI(
        # TODO: Make sure this API key is set in the environment variable (best not to hard code it)
        api_key=os.environ[api_key_name],
        base_url=base_url,
    )

def format_query_title_list(query_job_titles):
    output_string = ''
    for i, title in enumerate(query_job_titles):
        output_string += f'{i+1}. `{title}`\n'
    return output_string
    
def generate_prompt(query_job_titles : list[str] | np.ndarray, num_examples_per_title=5):
    # Spoofing the assistant response to encourage a certain format
    return [
        # {"role": "system", "content": "You are an expert in recruitment, staffing and HR."},
        {"role": "user", "content": f"{base_prompt.format(format_query_title_list(example_query_titles))}"},
        {"role": "assistant", "content": example_assistant_response},
        {"role": "user", "content": f"{follow_up_prompt.format(num_examples_per_title, format_query_title_list(query_job_titles))}"},
    ]

from openai import OpenAIError, RateLimitError, APIError, APIConnectionError

# Keep your existing imports and setup code...

async def async_make_api_call(client, model_name, messages, perturbation_std=0.0):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stop=["<query>"],
            # temperature=0.7 + rng.normal(0, perturbation_std)
        )
        return response, None
    except Exception as e:
        return None, e

async def async_main_stubborn(all_query_titles, client, model_name, output_path=None, chunk_size=5, 
                             num_examples_per_title=5, delay=2, giveup_after=10, 
                             max_backoff=60, initial_backoff=5):
    """
    Process job titles with exponential backoff for rate limits.
    
    Parameters:
    - all_query_titles: List of job titles to process
    - client: API client
    - model_name: Model to use
    - output_path: Where to save results
    - chunk_size: Number of titles to process in each batch
    - num_examples_per_title: Examples to generate per title
    - delay: Base delay between requests
    - giveup_after: Max attempts per chunk
    - max_backoff: Maximum backoff time in seconds
    - initial_backoff: Initial backoff time in seconds
    """
    responses_dict = {}
    
    # Load existing progress if output file exists
    if output_path and os.path.exists(output_path):
        try:
            with open(output_path, 'rb') as f:
                responses_dict = pickle.load(f)
            print(f"Loaded {len(responses_dict)} existing responses. Continuing from where we left off.")
        except Exception as e:
            print(f"Error loading existing progress: {e}")
    
    # Track remaining titles that haven't been processed yet
    remaining_titles = [title for title in all_query_titles if title not in responses_dict]
    
    # Process in chunks
    for i in trange(0, len(remaining_titles), chunk_size):
        current_query_titles = remaining_titles[i:i+chunk_size]
        attempts = 0
        current_backoff = initial_backoff
        
        while attempts < giveup_after:
            if attempts > 0:
                print(f'Attempt {attempts+1}/{giveup_after} for chunk {i//chunk_size + 1}/{len(remaining_titles)//chunk_size + 1}')
            
            # Create the API call task
            api_task = asyncio.create_task(
                async_make_api_call(
                    client,
                    model_name,
                    generate_prompt(
                        query_job_titles=current_query_titles,
                        num_examples_per_title=num_examples_per_title
                    ),
                    perturbation_std=0.1
                )
            )
            
            # Wait for the specified delay
            await asyncio.sleep(delay)
            
            # Wait for the API call to complete
            response, exception = await api_task
            
            # If we got a rate limit error, back off and retry
            if exception is not None:
                attempts += 1
                
                if isinstance(exception, (RateLimitError, APIError, OpenAIError, APIConnectionError)):
                    # Add jitter to avoid thundering herd problem
                    jitter = random.uniform(0.8, 1.2)
                    sleep_time = min(current_backoff * jitter, max_backoff)
                    
                    print(f"Rate limit hit. Backing off for {sleep_time:.2f} seconds...")
                    await asyncio.sleep(sleep_time)
                    
                    # Exponential backoff
                    current_backoff = min(current_backoff * 2, max_backoff)
                    continue
                else:
                    print(f"Error: {type(exception).__name__}: {exception}")
                    # For non-rate limit errors, use a smaller backoff
                    await asyncio.sleep(delay * 2)
                    continue
            
            # If we have a valid response, try to parse it
            try:
                parsed_response = parse_gpt_response(
                    gpt_output=response.choices[0].message.content,
                    num_query_titles=len(current_query_titles),
                    num_examples_per_query_title=num_examples_per_title,
                    throw_exception_on_failure=False
                )
                
                if parsed_response:
                    for query_title, response_list in zip(current_query_titles, parsed_response):
                        responses_dict[query_title] = response_list
                    
                    # Save progress after each successful chunk
                    if output_path:
                        with open(output_path, 'wb') as f:
                            pickle.dump(responses_dict, f)
                    
                    # Reset backoff on success
                    current_backoff = initial_backoff
                    break
                elif attempts > 1:
                    print('-------------------------------')
                    print('Output (parsing failed):')
                    print(response.choices[0].message.content)
                    # Continue with next attempt
            except Exception as e:
                print(f"Error parsing response: {e}")
            
            attempts += 1
        
        # Add a small delay between chunks to avoid rate limits
        if i + chunk_size < len(remaining_titles):
            await asyncio.sleep(delay)

    return responses_dict