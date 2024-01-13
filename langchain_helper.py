from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import os
os.environ['OPENAI_API_KEY'] = 'your api key'

llm = OpenAI(temperature=0.7)


def generate_restaurant_name_and_items(cuisine):
    # chain 1 -- Restuarant Name

    prompt_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restuarant for {cuisine} food . Suggest one name for it"
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_name, output_key="restaurant")

    # chain 2 -- menu items

    prompt_items = PromptTemplate(
        input_variables=['restaurant'],
        template="""Suggest some menu items for {restaurant} . return it as a comma separated list"""
    )
    food_chain = LLMChain(llm=llm, prompt=prompt_items,
                          output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant', "menu_items"]
    )

    response = chain({'cuisine': cuisine})

    return response


if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Indian"))
