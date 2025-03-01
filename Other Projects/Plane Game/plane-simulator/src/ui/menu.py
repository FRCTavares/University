from pygame import menu

class Menu:
    def __init__(self):
        self.options = ["Start Game", "Options", "Exit"]
        self.selected_option = 0

    def display_menu(self):
        print("Main Menu")
        for index, option in enumerate(self.options):
            if index == self.selected_option:
                print(f"> {option} <")
            else:
                print(option)

    def navigate(self, direction):
        if direction == "down":
            self.selected_option = (self.selected_option + 1) % len(self.options)
        elif direction == "up":
            self.selected_option = (self.selected_option - 1) % len(self.options)

    def select_option(self):
        if self.selected_option == 0:
            return "start_game"
        elif self.selected_option == 1:
            return "options"
        elif self.selected_option == 2:
            return "exit"