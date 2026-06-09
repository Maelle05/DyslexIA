import random

PASSAGES = [
    "Dans une forêt couverte de neige, un jeune renard avançait avec prudence entre les arbres. La nuit avait été froide et le sol brillait sous la lumière du matin. Le renard aimait explorer les endroits qu’il ne connaissait pas encore. Ce jour-là, il remarqua une série de traces qui traversaient une petite clairière. Curieux, il décida de les suivre. Les empreintes passaient près d’un ruisseau gelé puis contournaient un vieux sapin. Après un long moment, il découvrit un hérisson installé sous un tas de feuilles sèches. Le hérisson dormait profondément pour passer l’hiver au chaud. Le renard observa l’animal quelques instants puis s’éloigna sans faire de bruit. En rentrant vers son terrier, il comprit que la forêt cachait encore beaucoup de secrets et que chaque promenade pouvait lui apprendre quelque chose de nouveau.",
    "Le renard connaissait bien son territoire, mais l’hiver transformait tout. Les chemins habituels disparaissaient sous la neige et les odeurs étaient plus difficiles à suivre. Un matin, il aperçut un oiseau rouge posé sur une branche basse. L’oiseau semblait chercher de la nourriture. Le renard le regarda voler jusqu’à un buisson couvert de givre. Intrigué, il s’approcha et découvrit quelques baies oubliées. Il n’en avait jamais vu à cet endroit. Pendant plusieurs jours, il revint observer les oiseaux qui venaient s’y nourrir. Peu à peu, il apprit quels arbres gardaient leurs fruits même pendant la saison froide. Cette découverte lui permit de mieux comprendre la forêt. Chaque animal semblait connaître une astuce pour traverser l’hiver. Le renard était fier d’avoir appris quelque chose simplement en prenant le temps de regarder autour de lui.",
    "Le soleil se levait doucement sur la ville au bord de la mer. Les rues étaient encore calmes et peu de personnes étaient réveillées. Une jeune fille ouvrit sa fenêtre et respira l’air frais venu du large. Au loin, elle entendait les mouettes qui tournaient au-dessus du port. Les pêcheurs préparaient déjà leurs bateaux tandis que les premières lueurs du jour se reflétaient sur l’eau. En marchant vers la plage, elle remarqua que le sable gardait les traces de la marée de la nuit. Quelques coquillages brillants étaient restés près du rivage. Elle en ramassa un et le glissa dans sa poche comme souvenir. Ce premier matin lui semblait spécial. Tout paraissait paisible et nouveau. Elle eut l’impression que la ville se réveillait lentement autour d’elle, prête à commencer une nouvelle journée.",
    "C’était le premier jour de vacances dans une ville située au bord de la mer. Un garçon se leva très tôt pour observer le lever du soleil. Il descendit jusqu’au port où les commerçants installaient leurs étals. Certains vendaient du poisson frais, d’autres des fruits ou des fleurs. Le garçon aimait écouter les bruits du matin. Il entendait les cordes des bateaux qui bougeaient doucement contre les quais. Une vieille dame lui indiqua un endroit parfait pour admirer l’horizon. Depuis ce point, il vit le soleil sortir lentement de l’eau. Le ciel passa du gris au rose puis à l’orange. Le spectacle ne dura que quelques minutes mais resta longtemps dans sa mémoire. En retournant chez lui, il se promit de revenir le lendemain pour revoir ce moment magnifique.",
    "Léa voulait apprendre à faire du vélo sans les petites roues. Au début, elle avait un peu peur de tomber. Son père l’emmena dans un grand parc où les allées étaient larges et tranquilles. Il lui expliqua comment garder l’équilibre en regardant devant elle. Les premiers essais furent difficiles. Le vélo avançait de quelques mètres puis penchait d’un côté. Pourtant, Léa ne se découragea pas. Elle recommença encore et encore. Peu à peu, elle sentit que le guidon devenait plus stable. Un moment arriva où elle roula seule sans s’en rendre compte. Son père, resté quelques mètres derrière, l’applaudit avec enthousiasme. Quand elle réalisa qu’elle pédalait sans aide, un grand sourire apparut sur son visage. Elle venait d’apprendre quelque chose qui lui semblait impossible quelques heures plus tôt.",
    "Un samedi matin, Tom décida de s’entraîner à vélo sur la place du village. Plusieurs enfants jouaient déjà autour de lui. Sa grand-mère observait la scène depuis un banc. Tom savait pédaler mais il avait encore du mal à tourner correctement. Chaque fois qu’il voulait changer de direction, il ralentissait beaucoup. Sa grand-mère lui donna un conseil simple : rester calme et regarder l’endroit où il voulait aller. Tom suivit cette recommandation. Après quelques essais, ses virages devinrent plus fluides. Il réussit même à contourner une fontaine sans poser le pied par terre. Les autres enfants l’encouragèrent. À la fin de la matinée, il se sentait beaucoup plus confiant.",
    "Au fond d’une rue ancienne se trouvait l’atelier d’un vieil horloger. Chaque matin, il ouvrait ses volets et allumait une petite lampe au-dessus de son établi. Autour de lui, des dizaines d’horloges indiquaient des heures différentes. Certaines étaient très anciennes, d’autres plus récentes. L’horloger réparait avec patience les mécanismes délicats. Un jour, une petite fille lui apporta une montre qui appartenait à son grand-père. L’objet ne fonctionnait plus depuis longtemps. L’horloger examina les pièces avec attention puis commença son travail. Finalement, la montre recommença à avancer.",
]

QUESTIONS = [
    ["Que découvre le renard à la fin de sa recherche ?", "Un lapin qui court", "Une maison abandonnée", "Un hérisson endormi", 2],
    ["Qu’est-ce que le renard découvre près du buisson ?", "Des baies", "Un terrier", "Des champignons", 0],
    ["Que ramasse la jeune fille sur la plage ?", "Une bouteille", "Un coquillage", "Une étoile de mer", 1],
    ["Où le garçon observe-t-il le lever du soleil ?", "Depuis une montagne", "Depuis un endroit près du port", "Depuis une forêt", 1],
    ["Qui accompagne Léa au parc ?", "Son professeur", "Son père", "Son frère", 1],
    ["Quel conseil reçoit Tom ?", "Fermer les yeux", "Pédaler plus vite", "Regarder où il veut aller", 2],
    ["Que répare l’horloger ?", "Une montre", "Un vélo", "Un jouet", 0],
]



def get_passage():
    idx = random.randint(0, len(PASSAGES) - 1)

    text = PASSAGES[idx]
    q = QUESTIONS[idx]

    return {
        'text': text,
        'query': {
            'q': q[0],
            'a1': q[1],
            'a2': q[2],
            'a3': q[3],
            'valid': q[4],
        }
    }
