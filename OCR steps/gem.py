import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

# 1. Define Strict Structural Schema Maps
class QuestionItem(BaseModel):
    question_number: str = Field(description="The question label, e.g., '1', '2a', '3'")
    subject: str = Field(description="Strictly identify if it is Tamil, English, or Mathematics")
    main_topic: str = Field(description="The primary concept tested by the question")
    student_answer_transcription: str = Field(description="Full paragraph text in native script or LaTeX math notation")
    teacher_marks_assigned: int = Field(description="The numerical mark written by the teacher")
    max_marks_possible: int = Field(description="The maximum points possible if visible, otherwise provide a reasonable guess")

class ExamPaperPayload(BaseModel):
    page_contains_valid_test: bool = Field(description="True if this image contains a visible student test layout")
    questions: List[QuestionItem]

def extract_exam_data_via_ai(student_text_path, teacher_marks_path):
    # Ensure your correct secret key is dropped inside the quotes below
    API_KEY = "Gemini-API-KEY"
    
    # Initialize the modern 2026 client structure
    client = genai.Client(api_key=API_KEY)

    # 2. Bypassing Windows file locks (PIL OSError Fix)
    # Open the images as raw context bytes rather than streaming live filesystem pointers
    try:
        with open(student_text_path, "rb") as f:
            student_bytes = f.read()
        with open(teacher_marks_path, "rb") as f:
            teacher_bytes = f.read()
    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR]: Could not find your image files on disk: {e}")
        print("Ensure you ran the OpenCV preprocessing block first and the files exist in your folder.")
        return

    # Wrap raw inputs into explicit Part payloads for the v1/v1beta endpoint wrapper
    student_part = types.Part.from_bytes(data=student_bytes, mime_type="image/jpeg")
    teacher_part = types.Part.from_bytes(data=teacher_bytes, mime_type="image/jpeg")

    prompt = """
    You are an advanced exam processing system specializing in handwritten grading analysis.
    You are given two images of the exact same exam paper page:
    Image 1 (student_part): The student's handwriting (with the teacher's red marks removed).
    Image 2 (teacher_part): The teacher's red grading marks isolated on a white background.

    Your task is to geometrically align these images mentally and extract structured data.
    
    CRITICAL RULES:
    1. Parse the page section by section. Identify where a question block starts, where the student's answer text lies, and what score the teacher wrote nearby or over that answer.
    2. Read handwritten text accurately. The paper can contain English, Tamil script, or Mathematical equations. 
    3. Convert any mathematical formulas or structured math equations into clear LaTeX format (e.g. $$\\frac{x}{2}$$).
    4. Maintain the Tamil text in its native script formatting perfectly. Do not translate it.
    5. Output the final data strictly matching the requested JSON structure.
    """

    print("Sending processing request to Gemini Cluster...")
    
    try:
        # Route through current model blocks using memory segments
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[student_part, teacher_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExamPaperPayload,
                temperature=0.1
            ),
        )

        print("\n--- Processed Exam Data Successfully ---")
        print(response.text)
        
        # Save output structured file natively using explicit UTF-8 rules
        with open("extracted_student_data.json", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("\nSaved output mapping to: extracted_student_data.json")

    except Exception as error:
        print(f"\nAn error occurred during API execution: {error}")

# Run execution block
if __name__ == "__main__":
    # Ensure these point to the actual images created by your OpenCV step
    extract_exam_data_via_ai("clean_student_text.jpg", "isolated_teacher_marks.jpg")
