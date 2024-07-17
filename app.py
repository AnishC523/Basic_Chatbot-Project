# file for application gui

from tkinter import*
from chat import get_response, bot_name


# GUI COLORS & FONTS
BACKGROUND_COLOR = "#17202A"
BACKGROUND_GRAY = "#ABB2B9"
TEXT_COLOR = "#EAECEE"
FONT = "Helvetica 14"
BOLD_FONT = "Helvetica 13 bold"

# Application
class ChatApp:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    # main window properties
    def _setup_main_window(self):
        self.window.title("Chatbot")
        self.window.resizable(width = FALSE, height = FALSE) #keep window dimensions from changing
        self.window.configure(width = 470, height = 550, bg = BACKGROUND_COLOR)

        # head label
        head_label = Label(self.window, bg=BACKGROUND_COLOR, fg=TEXT_COLOR, text="WELCOME!",font=BOLD_FONT, pady=8)
        head_label.place(relwidth=1)
        # divider
        line = Label(self.window, width=40, bg=BACKGROUND_GRAY)
        line.place(relwidth = 1, rely=0.07, relheight=0.012)
        # text widget as instance variable
        self.text_widget = Text(self.window, width=20, height=2, bg = BACKGROUND_COLOR, fg=TEXT_COLOR, font = FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely = 0.08)
        self.text_widget.configure(cursor="arrow", state = DISABLED)
        # scroll bar 
        scrollbar = Scrollbar(self.text_widget) # inside text widget not self window
        scrollbar.place(relheight=1, relx=0.975)
        scrollbar.configure(command = self.text_widget.yview) # allows y position of text_widget to be modified (scroll)
        # bottom label
        bottom_label = Label(self.window, bg=BACKGROUND_GRAY, height=80)
        bottom_label.place (relwidth=1, rely = 0.825)
        # text entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font = FONT)
        self.msg_entry.place(relwidth = 0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus() # makes it so this widget is already selected
        self.msg_entry.bind("<Return>", self._on_enter)
        # send button
        send_button = Button(bottom_label, text="SEND", font=BOLD_FONT, width=20, bg=BACKGROUND_GRAY, command=lambda: self._on_enter(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)



    # function for when message is sent
    def _on_enter(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        # empty message
        if not msg:
            return

        # delete what is in text box
        self.msg_entry.delete(0,END)
        # message to be displayed
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state = NORMAL) #enable editing for text widget
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state = DISABLED) #disable
        # get chatbot response
        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state = NORMAL) 
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state = DISABLED) 

        # automatic scroll if window is full
        self.text_widget.see(END)

    def run(self):
        self.window.mainloop()


# main
if __name__ == "__main__":
    app = ChatApp()
    app.run()
