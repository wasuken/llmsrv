from dotenv import load_dotenv
import requests
import json
import os
from fastapi import FastAPI, Request, BackgroundTasks, Header
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)
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

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Ollamaの設定
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/elyza/Llama-3-ELYZA-JP-8B-GGUF:latest")

@app.post('/')
async def api_root():
    return {'message': "Healthy"}

@app.post("/callback")
async def callback(
    request: Request,
    background_tasks: BackgroundTasks,
    x_line_signature=Header(None),
    summary="LINE Message API Callback"
    ):
    # LINEからのリクエストを検証
    signature = request.headers['X-Line-Signature']
    body = await request.body()
    
    try:
        background_tasks.add_task(
            handler.handle, body.decode('utf-8'), x_line_signature,
        )
    except InvalidSignatureError:
        return {"error": "Invalid signature"}
    
    return {"message": "OK"}

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    with ApiClient(configuration) as api_clinet:
        line_bot_api = MessagingApi(api_clinet)

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
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=ai_response)]
                )
            )
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")  # デバッグ用にエラーを出力
            # エラー時は簡単なメッセージを返す
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="申し訳ありません、エラーが発生しました。")]
                )
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
