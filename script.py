import os
import zipfile
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Set your bot token here
BOT_TOKEN = "7604424348:AAHtkmD0YKApTg4B9-QeJNW5SXUmOmctK7E"

# Directory to store extracted files
EXTRACT_PATH = "extracted_files"

if not os.path.exists(EXTRACT_PATH):
    os.makedirs(EXTRACT_PATH)

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Send me a ZIP file, and I'll extract it for you!")

def handle_zip(update: Update, context: CallbackContext):
    file = update.message.document
    if file.mime_type == "application/zip":
        file_id = file.file_id
        new_file = context.bot.get_file(file_id)
        zip_path = os.path.join(EXTRACT_PATH, file.file_name)
        
        new_file.download(zip_path)

        # Extract the ZIP file
        extract_folder = os.path.join(EXTRACT_PATH, file.file_name.replace(".zip", ""))
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        # Get extracted file names
        extracted_files = os.listdir(extract_folder)
        extracted_list = "\n".join(extracted_files)

        update.message.reply_text(f"Extracted files:\n{extracted_list}")

        # Send extracted files
        for extracted_file in extracted_files:
            extracted_file_path = os.path.join(extract_folder, extracted_file)
            context.bot.send_document(chat_id=update.message.chat_id, document=open(extracted_file_path, 'rb'))

    else:
        update.message.reply_text("Please send a valid ZIP file.")

def main():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.document, handle_zip))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
