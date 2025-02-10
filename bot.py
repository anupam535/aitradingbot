import os
import asyncio
import requests
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Load environment variables
load_dotenv()

# Telegram Bot Token
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TEXT_TO_VIDEO_API_KEY = os.getenv('TEXT_TO_VIDEO_API_KEY')

# Global variables for user settings
user_settings = {}

# Text-to-Video AI API URL
TEXT_TO_VIDEO_API_URL = "https://api.text-to-video-service.com/generate"  # Replace with actual API URL

# Function to set video style
async def set_style(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    style = " ".join(context.args)
    if not style:
        await update.message.reply_text("Please specify a style. Example: /set_style anime")
        return
    if user_id not in user_settings:
        user_settings[user_id] = {}
    user_settings[user_id]['style'] = style
    await update.message.reply_text(f"Style set to: {style}")

# Function to set video quality
async def set_quality(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    quality = " ".join(context.args)
    if not quality:
        await update.message.reply_text("Please specify a quality. Example: /set_quality HD")
        return
    if quality.upper() not in ["HD", "4K"]:
        await update.message.reply_text("Invalid quality. Please choose 'HD' or '4K'.")
        return
    if user_id not in user_settings:
        user_settings[user_id] = {}
    user_settings[user_id]['quality'] = quality.upper()
    await update.message.reply_text(f"Quality set to: {quality.upper()}")

# Function to generate video
async def generate_video(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Please provide a prompt. Example: /generate A futuristic city at night")
        return

    # Get user settings
    settings = user_settings.get(user_id, {})
    style = settings.get('style', 'realistic')  # Default style
    quality = settings.get('quality', 'HD')  # Default quality

    # Prepare payload for the AI API
    payload = {
        "prompt": prompt,
        "style": style,
        "resolution": quality,
        "frame_rate": 30,  # Default frame rate
        "camera_angle": "wide",  # Default camera angle
        "lighting": "natural",  # Default lighting
        "special_effects": []  # Default no effects
    }

    headers = {
        "Authorization": f"Bearer {TEXT_TO_VIDEO_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        # Send request to the AI API
        response = requests.post(TEXT_TO_VIDEO_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            video_url = response.json().get("video_url")
            if video_url:
                await update.message.reply_text("Your video is ready!")
                await context.bot.send_video(chat_id=user_id, video=video_url)
            else:
                await update.message.reply_text("Failed to generate video. Please try again.")
        else:
            error_message = response.json().get("error", "Unknown error")
            await update.message.reply_text(f"Error: {error_message}")
    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")

# Start command
async def start(update: Update, context: CallbackContext):
    keyboard = [
        [InlineKeyboardButton("Set Style", callback_data="set_style")],
        [InlineKeyboardButton("Set Quality", callback_data="set_quality")],
        [InlineKeyboardButton("Generate Video", callback_data="generate")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Welcome! Use the buttons below to get started.", reply_markup=reply_markup)

# Error handler
async def error_handler(update: Update, context: CallbackContext):
    print(f"Update {update} caused error {context.error}")
    await update.message.reply_text("Oops! Something went wrong. Please try again.")

# Main function
if __name__ == '__main__':
    # Initialize the Telegram bot
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('set_style', set_style))
    application.add_handler(CommandHandler('set_quality', set_quality))
    application.add_handler(CommandHandler('generate', generate_video))

    # Add error handler
    application.add_error_handler(error_handler)

    # Start the bot
    print("Telegram bot started...")
    application.run_polling()
