
The repository is providing example projects with MCP servers.
The MCP servers are used for LLM integration, enabling LLM to perform
Monte-Carlo-simulation based tasks in scrience and engineering topics.


### Check '.env' file

After creating '.env' file according to the '.env.example' file, run the following
python script to test the validity of the "api key" for LLM.

The api key should be in **"OpenAI-compatible"** format. 

```
python test_api_key.py
```