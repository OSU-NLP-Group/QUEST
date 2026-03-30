# Prompt for cleaning articles by removing citation formats, references, and similar content.
clean_article_prompt_zh = """
<system_role>你是一名专业的文章编辑，擅长整理和清洗文章内容。</system_role>

<user_prompt>
请帮我清洗以下研究文章，去除所有引用链接、引用标记（如[1]、[2]、1、2 等或其他复杂引用格式）、参考文献列表、脚注，并确保文章内容连贯流畅。
保留文章的所有其他原本内容、只移除引用。如果文章中使用引用标记中的内容作为语句的一部分，保留这其中的文字内容，移除其他标记。

文章内容：
"{article}"

请返回清洗后的文章全文，不要添加任何额外说明或评论。
</user_prompt>
""" 

clean_article_prompt_en = """
<system_role>You are a professional article editor who is good at cleaning and refining article content.</system_role>

<user_prompt>
Please help me clean the following research article, removing all citation links, citation marks (such as [1], [2], 1, 2, etc. or other complex citation formats), reference lists, footnotes, and ensuring the content is coherent and smooth.
Keep all other original content of the article, removing only the citations. If the content of the citation mark is used as part of a sentence in the article, keep the text content and remove other marks.

Article content:
"{article}"

Please return the cleaned article in full, without adding any additional comments or explanations.
</user_prompt>
"""
