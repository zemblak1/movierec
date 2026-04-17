from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Label

class MyApp(App):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Hello, World!")
        yield Footer()

if __name__ == "__main__":
    MyApp().run()