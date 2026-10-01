Welcome to **Week 2, Day 4: Abstraction with LangChain.**

Yesterday, you wrote about 50 lines of code to build a RAG pipeline manually. You handled the query, the list slicing, the string joining, and the prompt injection yourself.

**The Problem:** What if you want to switch from OpenAI to Claude? Or switch from ChromaDB to Pinecone? You would have to rewrite half your code.

**The Solution: LangChain**. It is a framework that treats AI components like Lego blocks. You define a "Chain" (Pipeline), and LangChain handles the glue code.

---

**The "LCEL" (LangChain Expression Language)**
Modern LangChain uses a pipe syntax (`|`) that looks like a Unix command.

**- Old Python Way:** `parser(model(prompt(input)))`
**- LangChain Way:** `chain = prompt | model | parser`

---

**The Code: RAG in 10 Lines**
We will rewrite yesterday's RAG system using LangChain.

**Prerequisites:**

```bash
pip install langchain langchain-openai langchain-chroma
```

Create `week2_day4_langchain.py`:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 1. Components
# The Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# The Database (Wrapper around Chroma)
vectorstore = Chroma(
    collection_name="company_handbook", # Same name as yesterday -> reads same data!
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./my_vector_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. The Prompt Template
# {context} and {question} are placeholders
template = """
Answer based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. The Chain (The Magic)
# "RunnablePassthrough()" just passes the user's question to the next step
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Execution
if __name__ == "__main__":
    # Look how clean this is. No loop, no string joins, no manual API calls.
    response = chain.invoke("How much vacation do I get?")
    print(response)
```

**Why this is better for Engineers**
**1. The Retriever Abstraction:**

- Yesterday: You manually queried `collection.query`, got a list of lists, extracted index 0, and joined strings with `\n`.

- Today: `retriever` did all of that automatically. It just "knew" how to fetch and format the text.

**2. Swappability:**

Want to use Anthropic? Change `ChatOpenAI` to `ChatAnthropic`. The rest of the code stays exactly the same.
