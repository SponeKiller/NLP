from blessed import Terminal



term = Terminal()
        
with term.cbreak():
    # Starting index
    selected = 0

  
    
    options = {"Pretraing": train_model, 
                "Finetuning": finetune_model, 
                "Reinforce_learning": reinforce_model}
        
    
    # Event loop
    while True:
        print(term.move_yx(0, 0) + term.clear)
        print("Please select type of training\n")
        for index, option in enumerate(options):
            if index == selected:
                selected_option = option
                print(term.underline + option + term.no_underline)
            else:
                print(option)

        inp = term.inkey()
        
        if inp.is_sequence:
            if inp.name == 'KEY_UP':
                selected -= 1
            elif inp.name == 'KEY_DOWN':
                selected += 1
            elif inp.name == 'KEY_ENTER':
                break


        # Stay within the options list
        selected %= len(options)
        
print("\n")
print(f"{selected_option} module is loading, please wait.")

options[selected_option]()

