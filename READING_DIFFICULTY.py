import textstat

def analyze_text_difficulty(text):

    flesch_reading_ease = textstat.flesch_reading_ease(text)
    return flesch_reading_ease

text = 'However, hydrogels and nanoparticles systems have particular drawbacks that limit their widespread applications. In recent years, the incorporation of nanostructured systems into hydrogel has been developed as a novel way for the formation of new biomaterials with various functions to solve biomedical challenges.'
flesch = analyze_text_difficulty(text)

print(f"The Flesch Reading Difficulty score is: {100 - flesch}")

#Flesch Reading Ease: (let op: nog niet omgedraaid!)
    #Bereik: 0 tot 100.
    #Interpretatie:
    #90-100: Zeer eenvoudig, begrijpelijk voor een gemiddelde 11-jarige.
    #60-70: Standaardniveau, begrijpelijk voor 13- tot 15-jarigen.
    #0-30: Zeer moeilijk, universitair niveau.
    #Metric: De score is gebaseerd op de gemiddelde zinslengte (aantal woorden) en de gemiddelde woordlengte (aantal lettergrepen).

    #Belangrijk : ik heb de schaal omgedraaid !! EN eerst vertaald naar het Engels, dan terug!!