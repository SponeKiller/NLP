from typing import List, Tuple, TypedDict, Literal
from blessed import Terminal
import os
import time
from llm_api.llama_api import llama_api
from llm_api.gpt_api import gpt_api


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str

class Augmentation:
    """
    Augmentation makes the dataset bigger by adding new data to it.
    
    """
    
    
    def __init__(self, ds: List[Message]):
        """
        Initializes the augmentation class
        
        Args:
            ds (List[Message]): The dataset to be augmented 
            
        Returns:
            None
            
        Note:
            The dataset should be a list of dictionaries with the following keys:
            
            role: The role of the message (system, user, assistant)
            content: The content of the message
            
            To access augmented dataset, use the augmented_ds attribute
        """
        self.ds = ds
        self.augmented_ds = ds
        self._valid = True # Check if the instance is valid
        self._api_setting()
        self._terminal_UI()
        
        
    
    def synonymous_rep(self):
        """
        Synonymous replacement
        
        """
        if(self.is_valid == False):
            os.system('cls')
            print("Instance is no longer valid. Please create a new instance.")
            time.sleep(1)
            return
        
        
        instruction = "You will take the sentence and remove words without loosing the meaning. Write only output sentence."

        
        
    def random_insrt(self):
        """
        Random insertion 
        """
        if(self.is_valid == False):
            os.system('cls')
            print("Instance is no longer valid. Please create a new instance.")
            time.sleep(1)
            return
        
        
    def random_swap(self):
        """
        Random Swap 
        """
        if(self.is_valid == False):
            os.system('cls')
            print("Instance is no longer valid. Please create a new instance.")
            time.sleep(1)
            return
            
    def random_deletion(self):
        
        """
        Random deletion
        """
        
        if(self.is_valid == False):
            os.system('cls')
            print("Instance is no longer valid. Please create a new instance.")
            time.sleep(1)
            return
    def back_translation(self):
        """
        Back_translation - translation to different language and back
        """
        if(self.is_valid == False):
            os.system('cls')
            print("Instance is no longer valid. Please create a new instance.")
            time.sleep(1)
            return
    
    def _api_setting(self):
        """
        Setting api connection for augmentation
        
        """

        
        term = Terminal()
        while True:
            with term.hidden_cursor():
                
                with term.cbreak():
                    # Starting index
                    selected = 0

                    self.options = self._available_api() + ["Exit"]
                        
                    while True:
                        print(term.move_yx(0, 0) + term.clear)
                        print("Please select api for augmentation:")
                        print("Choose Exit to leave augmentation")
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
                                    self.is_valid = False
                                    os.system('cls')
                                    print("We are sorry to see you go. Goodbye!")
                                    time.sleep(1)
                                    return
                                else:
                                    break
                                
                        # Stay within the options list
                        selected %= len(self.options)
                        
            # Call the selected api              
            self.api = self.options[selected]()
            
            # Check if the api 
            if(self.api.is_valid == False):
                
                os.system('cls')
                print("API connection failed. Please try again.")
                time.sleep(1)
                
            else:
                
                break


                    
            
    def _available_api(self):
        """
        Returns the available api for augmentation
        
        Returns:
            List[str]: List of available api
        """
        if(self.is_valid == False):
            return
        
        return [llama_api, gpt_api]
        
        
    def _terminal_UI(self):
        """
        Gets from user info what type of augmentation he would like to do
        """
        if(self.is_valid == False):
            return
    
        term = Terminal()
        while True:
            with term.hidden_cursor():
                
                with term.cbreak():
                    # Starting index
                    selected = 0

                    self.options = self._available_training() + {"Exit": None}
                        
                    while True:
                        print(term.move_yx(0, 0) + term.clear)
                        print("Please select augmentation option:")
                        print("Choose Exit to leave augmentation")
                        for index, option in enumerate(self.options):
                            if index == selected:
                                self.selected_option = option[0]
                                print(term.underline + option[0] + term.no_underline)
                            else:
                                print(option[0])

                        inp = term.inkey()
                        if inp.is_sequence:
                            
                            if inp.name == "KEY_UP":
                                selected -= 1
                            elif inp.name == "KEY_DOWN":
                                selected += 1
                            elif inp.name == "KEY_ENTER":
                                
                                if (self.options[selected] == "Exit"):
                                    self.is_valid = False
                                    os.system('cls')
                                    print("We are sorry to see you go. Goodbye!")
                                    time.sleep(1)
                                    return
                                else:
                                    break
                                
                        # Stay within the options list
                        selected %= len(self.options)
                        
            # Calling augmentation method           
            self.options[selected]()
            
            with term.hidden_cursor():
                
                with term.cbreak():
                    # Starting index
                    selected = 0

                    self.options = ["Yes", "No"]
                        
                    while True:
                        print(term.move_yx(0, 0) + term.clear)
                        print("Would you like to continue in augmentation:")
                        
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
                                break
                                    
                                
                        # Stay within the options list
                        selected %= len(self.options)
                        
                if(self.options[selected] == "No"):
                    break
            
    
    def _available_training(self):
        """
        Returns the available training methods
        
        Returns:
            List[str]: List of available training methods
        """
        return  {
                    "Synonymous replacement": self.synonymous_rep, 
                    "Random insertion": self.random_insrt, 
                    "Random swap": self.random_swap, 
                    "Random deletion": self.random_deletion,
                    "Back translation": self.back_translation
                }
