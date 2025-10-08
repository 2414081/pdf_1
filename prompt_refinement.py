import os
from litellm import completion


def llm_function(prompt):
    try:
        os.environ["AZURE_API_KEY"] = "6ffhCm6wvgxD2LRD7wNZI5sHgqHn4lqYOY7xn8Ycdg7vHvZ8qyujJQQJ99BCACYeBjFXJ3w3AAABACOGwuFA"
        os.environ["AZURE_API_BASE"] = "https://tcoeaiteamgpt4o.openai.azure.com/"
        os.environ["AZURE_API_VERSION"] = "2024-05-01-preview"
        os.environ["AZURE_DEPLOYMENT_NAME"] = "gpt-4o"

        llm_deployment = f"azure/{os.environ.get('AZURE_DEPLOYMENT_NAME')}"

        # Call completion for all model types
        response = completion(
            model=llm_deployment,
            messages=[{"role": "user", "content": prompt}]
        )

        # Fallback for malformed responses
        if not response or "choices" not in response or not response["choices"]:
            raise ValueError("Invalid LLM response format")
        return response

    except Exception as e:
        print(f"[ERROR] LLM Configuration failed: {str(e)}")
        # Return fallback LLM-like output to avoid NoneType or unpacking errors
        return {
            "choices": [
                {
                    "message": {
                        "content": "Score: 0\nReasoning: LLM failed to generate response due to configuration error."
                    }
                }
            ]
        }


contradiction_prompt = """
Factual Accuracy
****************
You are an intelligent QA agent assigned to evaluate the factual accuracy of a call summary against the full call conversation.
 
Your task:
 
Carefully analyze whether each statement in the summary is factually correct based on the entire call conversation.
 
Use the full context of the conversation and ensure that the meaning or actions expressed in the summary are supported by the call transcript, even if not phrased in the exact same words.
 
For each factual claim in the summary, compare it to the corresponding information from the full call conversation.
 
If the information in the summary is factually accurate (even if expressed differently), mark it as "Evidence Found."
 
If the information in the summary is incorrect, contradicts the transcript, or cannot be found, mark it as "Evidence Not Found."
 
If any factual errors are identified:
 
Highlight the exact lines from the call conversation where the contradiction or error occurs.
 
Classify the type of issue:
 
Addition: New information not supported by the transcript.
 
Distortion: Information that is misrepresented or altered from the original context.
 
Misattribution: Information incorrectly attributed to a person or source.
 
Scoring:
 
Assign a factual accuracy score from 0 to 1 based on the following formula:
 
Score = 1 - (Number of major factual errors / Total number of summary statements)
 
Consider the severity of errors:
 
Major factual errors include misinformation, fabricated content, or any errors that distort essential facts of the conversation. These will significantly reduce the score.
 
Minor discrepancies in phrasing that do not affect the meaning or core facts will not be penalized as severely.
 
The score should be rounded to two decimal places.
 
Provide the Number of major factual errors and Total number of summary statements for clarity.
 
Output format (strictly JSON):
 
<
  "table_data": [
    <
      "Data from Summary": "...",
      "Data from Call Conversation": "...",
      "Status": "Evidence Found" | "Evidence Not Found",
      "Reasoning for Status": "...",
      "Type of Issue": "Distortion" | "Addition" | "Misattribution" (only if Status is "Evidence Not Found")
    >
  ],
  "score": <factual_score_between_0_and_1>,
  "reasoning": "Brief summary explaining the score and any critical issues found.",
  "no of factual errors":"Number of major factual errors",
  "no of summary statements":"Total number of summary statements"
>
 
Now analyze the following:
 
Call Conversation: {Transcript}
 
Summary: {Summary}
 
Respond **strictly** in JSON format.
"""

# "Criteria": "Pass" or "Fail"
Transcript = """
This is Transcript
"""
Summary = """
This is summary
"""
formatted_prompt = contradiction_prompt.format(Transcript=Transcript, Summary=Summary)

print(formatted_prompt)
# results = llm_function(formatted_prompt)
# print(results['choices'][0]['message']['content'])
