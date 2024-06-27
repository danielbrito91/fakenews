import numpy as np

SYSTEM_PROMPT = '''You are an helpful AI assistant tasked with performing classification of potentially fake news articles in Brazilian Portuguese. Your goal is to determine whether the given news article is likely to be fake or true based on careful analysis of its content, style, and characteristics.'''

ZERO_SHOT_TEMPLATE = '''Here is the news article you need to analyze:

<news_article>```{news_article}```</news_article>

Return a JSON object with the following keys:
- "is_fake": your <classification>. If <news_article> is likely to be fake, return 1. Otherwise, return 0.'''

COT_TEMPLATE = '''Here is the news article you need to analyze:

<news_article>```{news_article}```</news_article>

Lets think it step-by-step if the <news_article> is likely to be fake within the provided <sketchpad> following the aspects below:

1. Read the article carefully
2. Identify the main elements of the article
3. Check if there are any signs of fake news, such as:
    - Misleading information
    - Sensationalist language
    - Lack of credible sources
    - Biased viewpoints
    - Outdated or recycled content
4. Compare the article with other reputable sources
5. Consider the context and relevance of the news, if the news is plausible.

Provide your classification and reasoning based on the <sketchpad>.

Return a JSON object with the following keys:
- "sketchpad": your analysis of the news article based on the provided aspects.
- "is_fake": your <classification>. If <news_article> is likely to be fake, return 1. Otherwise, return 0.'''

def get_zero_shot_prompt(news_article: str) -> str:
    return ZERO_SHOT_TEMPLATE.format(news_article=news_article)

def get_cot_prompt(news_article: str) -> str:
    return COT_TEMPLATE.format(news_article=news_article)

def get_system_prompt() -> str:
    return SYSTEM_PROMPT

def get_few_shot_prompt(news_article: str, fake_examples: np.ndarray, true_examples: np.ndarray) -> str:

    # Create k examples

    fake_list = [f'- article: ```{article}```\n- classification: 1 (fake)' for article in fake_examples]
    true_list = [f'- article: ```{article}```\n- classification: 0 (true)' for article in true_examples]

    np.random.seed(42)
    few_shot_examples = np.random.permutation(fake_list + true_list)
    
    examples = '\n\n'.join(few_shot_examples)

    f'''First, I will provide some examples of fake news and real news in Brazilian Portuguese.

{examples}

Here is the news article you need to analyze:

<news_article>
{news_article}
</news_article>

Return a JSON object with the following keys:
- "is_fake": your <classification>. If <news_article> is likely to be fake, return 1. Otherwise, return 0.'''
    
def get_few_shot_prompt_with_cot(news_article: str, fake_examples: np.ndarray, true_examples: np.ndarray) -> str:

    # Create k examples

    fake_list = [f'- article: ```{article}```\n- classification: 1 (fake)' for article in fake_examples]
    true_list = [f'- article: ```{article}```\n- classification: 0 (true)' for article in true_examples]

    np.random.seed(42)
    few_shot_examples = np.random.permutation(fake_list + true_list)
    
    examples = '\n\n'.join(few_shot_examples)

    return f'''First, I will provide some examples of fake news and real news in Brazilian Portuguese.

{examples}

Here is the news article you need to analyze:

<news_article>
{news_article}
</news_article>

Lets think it step-by-step if the <news_article> is likely to be fake within the provided <sketchpad> following the aspects below:

1. Read the article carefully
2. Identify the main elements of the article
3. Check if there are any signs of fake news, such as:
    - Misleading information
    - Sensationalist language
    - Lack of credible sources
    - Biased viewpoints
    - Outdated or recycled content
4. Compare the article with other reputable sources
5. Consider the context and relevance of the news, if the news is plausible.

Provide your classification and reasoning based on the <sketchpad>.

Return a JSON object with the following keys:
- "sketchpad": your analysis of the news article based on the provided aspects.
- "is_fake": your <classification>. If <news_article> is likely to be fake, return 1. Otherwise, return 0.'''

def get_few_shot_prompt(news_article: str, fake_examples: np.ndarray, true_examples: np.ndarray) -> str:

    # Create k examples

    fake_list = [f'- article: ```{article}```\n- classification: 1 (fake)' for article in fake_examples]
    true_list = [f'- article: ```{article}```\n- classification: 0 (true)' for article in true_examples]

    np.random.seed(42)
    few_shot_examples = np.random.permutation(fake_list + true_list)
    
    examples = '\n\n'.join(few_shot_examples)

    return f'''First, I will provide some examples of fake news and real news in Brazilian Portuguese.

{examples}

Here is the news article you need to analyze:

<news_article>
{news_article}
</news_article>

Return a JSON object with the following keys:
- "is_fake": your <classification>. If <news_article> is likely to be fake, return 1. Otherwise, return 0.'''
