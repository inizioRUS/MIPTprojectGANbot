import os

import MlPart
from config import *
from telegram import ForceReply, Update, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, \
    ReplyKeyboardMarkup, ReplyKeyboardRemove, Bot
from telegram.ext import Application, Updater, CommandHandler, CallbackContext, MessageHandler, filters, \
    CallbackQueryHandler

keyboard = [

    [
        KeyboardButton('/help'),
        KeyboardButton('/transfer_style'),
        KeyboardButton('/cancel'),
    ]
]
reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


async def start(update: Update, context: CallbackContext) -> None:
    context.chat_data["what_do"] = 0

    user = update.effective_user
    await update.message.reply_html(
        rf"""Hi {user.mention_html()}!
        This is a transfer-stylebot that changes your photos as you want""",
        reply_markup=reply_markup
    )


async def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("""Help! Hello :)""")


async def fotoalysis(update: Update, context: CallbackContext) -> None:
    if context.chat_data["what_do"] == 1:
        context.chat_data["photo"] = await (
            await context.bot.get_file(update.message.photo[-1].file_id)).download_as_bytearray()
        context.chat_data["what_do"] = 2
        await update.message.reply_text("One more(style photo)", reply_markup=reply_markup)
    elif context.chat_data["what_do"] == 2:
        photo2 = await (
            await context.bot.get_file(update.message.photo[-1].file_id)).download_as_bytearray()
        ml = MlPart.MLPart()
        output = ml.run_style_transfer(ml.cnn_normalization_mean, ml.cnn_normalization_std,
                                       *ml.load_image(context.chat_data["photo"], photo2))
        await update.message.reply_photo(output)
        await update.message.reply_text("Its all bro", reply_markup=reply_markup)


async def transfer_photo(update: Update, context: CallbackContext) -> None:
    context.chat_data["what_do"] = 1
    await update.message.reply_text("Send your first photo(content photo)", reply_markup=reply_markup)


async def cancel(update: Update, context: CallbackContext) -> None:
    context.chat_data["what_do"] = 0
    await update.message.reply_text("Okay", reply_markup=reply_markup)


def main() -> None:
    application = Application.builder().token(os.environ['TG_BOT_TOKEN']).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("Help", help_command))
    application.add_handler(CommandHandler("transfer_style", transfer_photo))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, fotoalysis))
    application.run_polling()


if __name__ == "__main__":
    main()
