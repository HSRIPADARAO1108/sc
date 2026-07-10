from patterns import patterns_binary, display_pattern


for name,pattern in patterns_binary.items():

    print("\nCharacter:",name)

    print(
        display_pattern(pattern)
    )
