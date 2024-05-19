import openai
import os
import time
from blessed import Terminal




class gpt_api:
    def __init__(self, augmentation = True)-> None:
        self.augmentation = augmentation
        self.is_valid = True
        self._api_setting()
        self._model_setting()
        
    def _api_setting(self):
            
        
        api_key = ""
        
        term = Terminal()
        with term.hidden_cursor():
            with term.cbreak():
                while True:
                    print(term.move_yx(0, 0) + term.clear)
                    print("Please insert your API key below and press Enter: ")
                    print(api_key)
                    inp = term.inkey()
                    if inp.is_sequence:
                        if inp.name == "KEY_ENTER":
                            break
                        elif inp.name == "KEY_BACKSPACE":
                            api_key = api_key[:-1]
                    else:
                        api_key += inp
        
        self.client = openai.OpenAI(api_key=api_key)
        os.system('cls')
        print("Waiting for connection...")
        time.sleep(1)
        self._connection_check()

    def _model_setting(self):
        
        if(self.is_valid == False):
            return
        
        term = Terminal()
        models = self.available_models()
        with term.hidden_cursor():
            with term.cbreak():
                # Starting index
                selected = 0

                self.options = models + ["Exit"]
                    
                while True:
                        print(term.move_yx(0, 0) + term.clear)
                        print("Welcome to GPT API")
                        print("Please select a model to use:\n")
                        for index, option in enumerate(self.options):
                            if index == selected:
                                self.selected_option = option
                                print(term.underline + option + term.no_underline)
                            else:
                                print(option)

                        inp = term.inkey()
                        if inp.is_sequence:
                            
                            if inp.name == "KEY_UP":
                                selected -= 1
                            elif inp.name == "KEY_DOWN":
                                selected += 1
                            elif inp.name == "KEY_ENTER":
                                
                                if (self.options[selected] == "Exit"):
                                    os.system('cls')
                                    print("We are sorry to see you go. Goodbye!")
                                    time.sleep(1)
                                    self.is_valid = False
                                    return
                                else:
                                    break
                                
                        # Stay within the options list
                        selected %= len(self.options)
        
        self.model = self.options[selected]
        
        if(self.augmentation == False):
            self._terminal_UI()
        
    def query(self, query: str ):
        
        if(self.is_valid == False):
            return print("Instance is no longer valid. Please create a new instance.")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ]
        )
        return response.choices[0].message.content
    
    def _connection_check(self):
        
        """
        Check if the connection to the API is working
        
        """
        
        try:
            self.client.chat.completions.create(model = self.available_models()[0], messages = [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": "Hello"}
            ]
            )
            os.system('cls')
            print("Connection was successful")
            time.sleep(1)
            
            
        except Exception as e:
            os.system('cls')
            print("Connection error")

            self.is_valid = False
            return
    
    def _terminal_UI(self):
        
        if(self.is_valid == False):
            return
        
        term = Terminal()
        print("Please enter your query below:")
        print("Press Esc to exit")
        
        with term.hidden_cursor():
            with term.cbreak():
                while True:
                    query = ""    
                    while True:
                        print(term.move_yx(0, 0) + term.clear)
                        print("User input:")
                        print(query)

                        inp = term.inkey()
                        
                        if inp.is_sequence:
                            if inp.name == "KEY_ENTER":
                                break
                            elif inp.name == "KEY_BACKSPACE":
                                query = query[:-1]
                            elif inp.name == "KEY_ESCAPE":
                                print(term.move_yx(0, 0) + term.clear)
                                print("We are sorry to see you go. Goodbye!")
                                return
                        else:
                            query += inp
                        
                    print(term.move_yx(0, 0) + term.clear)
                    print("Please wait...")
                    if(self.query(query)):
                        print(term.move_yx(0, 0) + term.clear)
                    print(self.query(query))
                    time.sleep(1)
                    print("Press Enter key to continue")
                    print("Press Esc to exit")
                    while True:
                        inp = term.inkey()
                        if inp.is_sequence:
                            if inp.name == "KEY_ENTER":
                                break
                            elif inp.name == "KEY_ESCAPE":
                                print(term.move_yx(0, 0) + term.clear)
                                print("We are sorry to see you go. Goodbye!")
                                time.sleep(1)
                                return
                    
                  
                
    def available_models(self) -> list[str]:
        """
        Available models for the API
        
        Note: Models are static because the API does not provide a way to get the available models 
        
        """
        return [
            "gpt-3.5-turbo-0125",
            "gpt-4",
            "gpt-4o"
        ]


# if program is run directly , we will run chat function
if __name__ == "__main__":
    gpt_api(False)

