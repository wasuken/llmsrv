from fastapi import FastAPI, Request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
import requests
import json
import os

# 環境変数の読み込み
load_dotenv()

app = FastAPI()

# LINE Bot設定（.envから読み込み）
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# 設定値のチェック
if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    raise ValueError(
        "LINE_CHANNEL_SECRET と LINE_CHANNEL_ACCESS_TOKEN を .env に設定してください！"
    )

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Ollamaの設定
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/elyza/Llama-3-ELYZA-JP-8B-GGUF:latest")

@app.post("/webhook")
async def webhook(request: Request):
    # LINEからのリクエストを検証
    signature = request.headers['X-Line-Signature']
    body = await request.body()
    
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        return {"error": "Invalid signature"}
    
    return {"message": "OK"}

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    
    # システムプロンプトを設定
    system_prompt = """あなたは親しみやすく、でも丁寧な日本語で会話するアシスタントです。
    ユーザーの質問に対して、分かりやすく具体的に回答してください。
    必要に応じて、例を交えながら説明することもできます。"""
    
    # Ollamaへのリクエスト
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_prompt}\n\nユーザー: {user_message}\nアシスタント: ",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "num_predict": 1000
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        ai_response = response.json()["response"]
        
        # LINEへ返信
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=ai_response)
        )
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")  # デバッグ用にエラーを出力
        # エラー時は簡単なメッセージを返す
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="申し訳ありません、エラーが発生しました。")
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
