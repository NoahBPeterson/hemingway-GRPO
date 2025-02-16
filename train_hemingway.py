from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
PatchFastRL("GRPO", FastLanguageModel)

import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from hemingway import analyze_text
import sys
from s1_grpo_trainer import MyS1GRPOTrainer
import wandb

model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer2 = AutoTokenizer.from_pretrained(model_name)

# %% [markdown]
# ## System Prompt and Writing Prompts
SYSTEM_PROMPT = """
You are an expert writer who specializes in Hemingway-style clear, concise communication.
Write in a direct, active voice. Avoid unnecessary words, qualifiers, and complex phrases.
Keep sentences short and impactful, but understand that you're writing a whole story from a prompt.
Your writing should be clear enough for anyone to understand, and long enough to flesh out a story.
Once again, as long of a story as you can write.
"""

WRITING_PROMPTS = [
    "A lighthouse keeper finds a message in a bottle warning of an event 100 years past. Write the next tide.",
    "A tattoo artist discovers their ink alters clients' memories. Write their moral crisis during a mobster's appointment.",
    "All newborns stop crying. Write a pediatrician's journal entry on the day the first silent scream appears.",
    "A chef must prepare a last meal using ingredients representing their estranged daughter's life. Write the kitchen reckoning.",
    "Photographs now capture people's secrets instead of smiles. Write a divorce lawyer flipping through a client's wedding album.",
    "A town where shadows move independently at noon. Write a sunlit chase between a woman and her own silhouette.",
    "Every clock in the city stops at 3:07 AM. Write the overnight radio host's monologue as listeners vanish mid-call.",
    "A librarian must burn forbidden books to keep the library warm. Write their choice between saving Plato or a heathen cookbook.",
    "Soldiers find an ancient typewriter that makes anything typed become true. Write the private's single-sentence temptation.",
    "A child's imaginary friend appears at their parent's murder trial. Write the courtroom gasp when the friend testifies.",
    "All mirrors now show strangers. Write a makeup artist's confrontation with the face claiming to be her twin.",
    "A jazz musician's saxophone conjures literal storms. Write the club owner's ultimatum: stop playing or drown the city.",
    "Gravestones display causes of death for the living. Write a jogger reading 'Suffocated by Silence' on her own plaque.",
    "A florist's bouquets erase memories. Write a widow ordering roses to forget, then begging for remembrance.",
    "Firefighters battle blazes that burn secrets instead of wood. Write the captain rescuing diaries from her own home.",
    "A bar serves drinks that temporarily swap lives. Write the ex-con and judge toasting with traded identities.",
    "Children's drawings come to life at midnight. Write parents barricading doors against crayon monsters.",
    "A detective solves crimes by tasting the last meal of victims. Write their revulsion upon recognizing their mother's recipe.",
    "All actors must live their roles between performances. Write Macbeth's lead begging to play a comedy.",
    "A gardener grows plants that cure specific regrets. Write the moment they find a weed labeled 'Should Have Stayed.'",
    "A pianist's left hand plays the future, right hand the past. Write their concert where both hands strike the same note.",
    "Migratory birds now carry human memories south. Write a woman tying letters to geese heading toward her amnesic lover.",
    "A seamstress stitches lies into clothing. Write her panic when the mayor's truthful suit unravels during a scandal.",
    "First kisses erase themselves from memory. Write lovers meeting daily, forever experiencing new firsts.",
    "A taxidermist receives a creature no one has ever seen. Write their decision: stuff it or prove it existed.",
    "Clouds form shapes of viewers' deepest shames. Write a cloudless sky cult's collapse when a boy admits he sees nothing.",
    "A locksmith opens any door, including metaphorical ones. Write their fee for accessing a politician's buried conscience.",
    "All written words disappear at dawn. Write a poet's race against sunrise to memorize verses for their dying partner.",
    "A therapist treats patients' alternate-reality selves. Write the session where both versions demand opposing cures.",
    "A baker's bread determines consumers' emotions. Write the protest when they stop making 'Contentment' rye.",
    "A historian wakes speaking a dead language. Write their conversation with the only other speaker—a hospice patient.",
    "Paintings whisper insults to viewers. Write an art critic's five-star review of a portrait that called them a fraud.",
    "A mail carrier delivers letters to the deceased. Write the day a WWII soldier responds to his 104-year-old bride.",
    "Actors in a crime reenactment show begin committing real murders. Write the director's guilt-ridden ratings surge.",
    "A mechanic fixes cars involved in fatal crashes. Write their discovery of a vehicle that predates automobiles.",
    "A child inherits their imaginary friend's childhood home. Write the realtor's tour revealing claw marks and tiny doors.",
    "A beekeeper's hive produces honey that reveals truths. Write the town's chaos after the mayor samples 'Corruption' batch.",
    "All lies manifest as physical scars. Write a lawyer's closing argument with bleeding cheeks.",
    "A diver finds a skeleton holding modern tech in a 1800s shipwreck. Write the museum's cover-up meeting.",
    "A coroner hears cadavers' final thoughts. Write their breakdown when a body whispers today's date.",
    "A prank call connects to the caller's future self. Write the teen's reaction to hearing middle-aged laughter.",
    "A colorblind painter's works predict disasters. Write the gallery opening where buyers bid on impending tragedies.",
    "Library books now open to readers' unwritten memoirs. Write a patron burning their volume in the parking lot.",
    "A chef's food tastes like consumers' happiest memories. Write their despair when a critic says `It's bland.`",
    "A child's nightmare monster begs for asylum from worse terrors. Write the parents' midnight negotiation.",
    "A town celebrates opposite day yearly—crimes included. Write the banker's robbery during legalized lawlessness.",
    "A musician's compositions control weather. Write their final symphony to end a decade-long drought.",
    "A translator deciphers animal speech as existential rants. Write the zookeeper's crisis hearing elephants debate nihilism.",
    "A blacksmith forges weapons from clients' regrets. Write the warrior's shock when their sword whispers childhood shames.",
    "All dreams now occur in public parks. Write lovers meeting nightly at the bench where their subconsciouses collide.",
    "A translator deciphers animal speech as existential rants. Write the zookeeper's crisis hearing elephants debate nihilism.",
    "A blacksmith forges weapons from clients' regrets. Write the warrior's shock when their sword whispers childhood shames.",
    "All dreams now occur in public parks. Write lovers meeting nightly at the bench where their subconsciouses collide.",
    "A barista brews coffee that reveals drinkers' deepest fears. Write the regular who orders a double espresso and sees nothing.",
    "A fisherman catches a fish that grants one unspoken wish. Write the moment he realizes his silence has already cost him.",
    "A town where laughter is currency. Write the stand-up comedian bankrupting hecklers with a single joke.",
    "A tailor sews clothes that fit the wearer's soul, not their body. Write the CEO's panic when their suit shrinks to child size.",
    "A gardener grows flowers that bloom in the color of lies. Write the detective's shock at a black rose in their own garden.",
    "A musician's songs erase themselves after being heard. Write the concert where the audience forgets the encore mid-applause.",
    "A photographer captures images of people's futures. Write the subject who sees nothing but darkness in every frame.",
    "A librarian guards books that rewrite themselves based on readers' morals. Write the thief stealing a blank tome only to find it filled with their crimes.",
    "A chef cooks meals that taste like diners' happiest memories. Write the orphan who tastes nothing but salt.",
    "A clockmaker builds timepieces that count down to viewers' deaths. Write the moment a clock stops at zero but its owner lives.",
    "A painter's portraits age while the subjects stay young. Write the model demanding their youth back from the canvas.",
    "A town where shadows move independently. Write the man chasing his shadow through a moonless night.",
    "A scientist invents a machine that translates silence. Write the first message from a mute child's unspoken words.",
    "A gravedigger hears the dead whispering through the soil. Write the day they recognize their own voice below.",
    "A baker's bread causes consumers to relive their worst memories. Write the town's addiction to the bitter loaves.",
    "A cobbler makes shoes that walk wearers into alternate realities. Write the child who steps into a world where they were never born.",
    "A florist's bouquets bloom in the color of unspoken love. Write the widow receiving a black rose from her late husband.",
    "A town where rain falls only on the guilty. Write the priest's sermon during a drought that spares no one.",
    "A puppeteer's marionettes act out viewers' hidden desires. Write the audience's horror when the puppets refuse to stop.",
    "A blacksmith forges keys that unlock any door, including metaphorical ones. Write the thief unlocking their own heart by accident.",
    "A musician's violin plays the songs of the dead. Write the widow who hears her husband's unfinished symphony.",
    "A lighthouse keeper tends a beacon that guides lost souls. Write the night the light goes out during a storm.",
    "A poet's verses come true when spoken aloud. Write the bard's terror after accidentally cursing their lover.",
    "A town where children are born with the memories of their ancestors. Write the parents raising a child who remembers their own death.",
    "A sculptor's statues come to life at midnight. Write the artist's confrontation with their masterpiece.",
    "A fortune teller's predictions are always wrong—until they aren't. Write the client who bets their life on a lie.",
    "A town where everyone hears their own thoughts in others' voices. Write the man who hears his enemy's voice in his head.",
    "A chef's dishes erase diners' memories. Write the couple sharing a meal to forget, only to fall in love again.",
    "A clockmaker builds a timepiece that runs backward. Write the town aging in reverse as the clock ticks.",
    "A painter's landscapes change based on viewers' moods. Write the critic who sees only storms in every canvas.",
    "A town where mirrors reflect the viewer's future. Write the woman who sees only darkness in her reflection.",
    "A musician's songs heal physical wounds but scar the soul. Write the soldier choosing between pain and guilt.",
    "A blacksmith forges swords that bleed when wielded by the unworthy. Write the knight's shame as his blade weeps.",
    "A librarian guards books that can only be read once. Write the scholar who burns their favorite tome to save its words.",
    "A town where silence is deadly. Write the mute child who becomes a hero.",
    "A photographer captures images of people's fears. Write the subject who sees their own face in every frame.",
    "A gardener grows plants that bloom in the color of grief. Write the widow who tends a garden of black roses.",
    "A sailor steers a ship through a storm of memories. Write the captain who faces the ghost of his past.",
    "A writer finds pages of a future novel in their mailbox. Write the moment the story begins to change reality.",
    "A painter sees the world in shades of regret. Write the portrait that refuses to reflect happiness.",
    "A cobbler mends shoes that have walked through dreams. Write the journey of a pair that outlasts their owner.",
    "A clocktower chimes a secret every midnight. Write the man who listens for answers in the echoes.",
    "A street performer steals moments from strangers. Write the day the thief is caught in his own act.",
    "A gardener cultivates a rare flower that blooms only at sorrow's peak. Write the story of a heart revived by its petals.",
    "A doctor heals wounds that were never meant to close. Write the operating room where silence screams.",
    "A bartender mixes cocktails that stir forgotten truths. Write the confession that escapes between ice and glass.",
    "A detective investigates crimes that exist only in whispered dreams. Write the case that blurs the line between memory and myth.",
    "A teacher discovers lessons etched in the scars of old buildings. Write the student who deciphers history from stone.",
    "A sculptor carves statues that mourn the lost. Write the moment when stone weeps for its creator.",
    "A musician plays tunes that erase the sound of regrets. Write the note that mutes a lifetime of sorrow.",
    "A librarian archives stories that have never been told. Write the day a forbidden tale finds its voice.",
    "A farmer harvests crops that echo the seasons of love. Write the festival where hearts replant hope.",
    "A tailor stitches fabrics from the threads of forgotten dreams. Write the suit that carries a man's silent aspirations.",
    "A chef cooks meals that taste like distant memories. Write the diner who discovers the flavor of home in every bite.",
    "A fisherman casts nets to catch fleeting moments. Write the haul that reveals a glimpse of eternity.",
    "A mountaineer climbs peaks that mirror the soul's ascent. Write the summit where clouds embrace lost ambitions.",
    "A miner digs through layers of buried time. Write the cavern where echoes of the past speak clearly.",
    "A pilot navigates skies that reflect inner turbulence. Write the flight where storms mirror personal battles.",
    "A seamstress weaves garments from strands of destiny. Write the dress that changes the wearer's fate.",
    "A merchant trades relics of forgotten eras. Write the sale that exchanges a lifetime of memories.",
    "A hermit writes letters addressed to the wind. Write the reply that carries echoes of long-lost love.",
    "A gambler bets with time itself. Write the hand that risks tomorrow on a single heartbeat.",
    "A forger creates art that captures invisible truths. Write the unveiling of a masterpiece that betrays its maker.",
    "A florist arranges bouquets that fade into whispered regrets. Write the arrangement that leaves a lingering scent of longing.",
    "A monk chants prayers that dissolve into the night. Write the ritual where silence transforms into a hymn.",
    "A librarian finds a book that writes itself with every reader's tear. Write the story that records unspoken grief.",
    "A blacksmith shapes iron into symbols of lost honor. Write the sword that bears the weight of forgotten oaths.",
    "A historian deciphers ruins humming with ancient sorrows. Write the discovery that resurrects buried truths.",
    "A traveler follows maps drawn by fate. Write the route that leads to a destiny written in the stars.",
    "A dancer moves with steps borrowed from the wind. Write the performance that unites hearts in silent understanding.",
    "A poet scribbles verses echoing in abandoned corridors. Write the stanza that awakens dormant memories.",
    "A locksmith crafts keys to unlock hidden selves. Write the door that opens to reveal a soul unburdened by lies.",
    "A sculptor molds clay that absorbs whispered confessions. Write the figure that unveils the artist's deepest secret.",
    "A botanist cultivates plants that hum ancient lullabies. Write the garden that sings of forgotten childhood dreams.",
    "A blacksmith forges chains that bind past misdeeds. Write the link that shatters under the force of redemption.",
    "A navigator charts courses by the stars of lost wishes. Write the journey where the night sky pens a new destiny.",
    "A chef bakes bread infused with silent hopes. Write the bite that rekindles a love thought lost.",
    "A storyteller recounts legends etched in scars. Write the myth that heals the fractures of a divided town.",
    "A carpenter constructs cradles for dreams yet to be born. Write the lullaby that rocks a broken world to sleep.",
    "A sailor drifts on tides of unspoken remorse. Write the voyage where the sea swallows forgotten sins.",
    "A poet finds verses in the patterns of falling rain. Write the sonnet that washes away a lifetime of regret.",
    "A gardener tends a labyrinth of reflective hedges. Write the maze that reveals truths hidden with every turn.",
    "A conductor directs an orchestra of silent instruments. Write the symphony that mutes the cacophony of doubts.",
    "A miner extracts gems that shimmer with captured memories. Write the moment a stone illuminates a dark past.",
    "A tailor cuts cloth with scissors of resolve. Write the garment that transforms sorrow into a fresh beginning.",
    "A wanderer collects fragments of broken time. Write the puzzle that pieces together a lost era.",
    "A singer belts notes that summon forgotten ancestors. Write the performance that bridges generations with a single melody.",
    "A thief steals whispers from the corridors of history. Write the heist where silence reveals a louder truth.",
    "A dancer spins in circles of recollection. Write the pirouette that unravels a tapestry of yesteryears.",
    "A baker crafts cakes that crumble like old promises. Write the slice that unveils the bittersweet flavor of truth.",
    "A farmer tills land that remembers ancient footsteps. Write the harvest where the soil confesses its secrets.",
    "A blacksmith hammers dreams into cold metal. Write the echo of the anvil as a new hope is forged.",
    "A writer scribbles on pages destined to vanish at dawn. Write the manuscript that clings to a fleeting moment.",
    "A jeweler polishes stones that mirror hidden sorrows. Write the gem that refracts a lifetime of unspoken words.",
    "A falconer trains birds to carry silent messages. Write the flight that delivers hope to a desolate heart.",
    "A wanderer maps routes through forgotten alleys. Write the path that leads to a rediscovered past.",
    "A photographer snaps shots of fleeting regrets. Write the picture that freezes an unuttered truth.",
    "A seamstress embroiders memories into faded fabric. Write the dress that tells a story of loss and renewal.",
    "A watchmaker repairs clocks that tick in tune with heartbeats. Write the moment when time stands still for love.",
    "A librarian archives secrets in tomes of the unremembered. Write the revelation that shatters decades of silence.",
    "A cartographer draws maps of imagined realms. Write the expedition that blurs the line between myth and reality.",
    "A sailor navigates waters that mirror the soul's depths. Write the journey where the ocean reveals a hidden self.",
    "A monk meditates on words unsaid and dreams unchronicled. Write the epiphany that dissolves the boundaries of time.",
    "A traveler collects echoes from the ruins of memory. Write the encounter where silence speaks louder than words.",
    "A sculptor carves a monument from sorrowful stone. Write the unveiling that awakens emotions long entombed.",
    "A chef simmers a broth that stews the flavors of the past. Write the bowl that comforts a soul weary from loss.",
    "A poet whispers verses to the midnight wind. Write the poem that carries a secret across endless darkness.",
    "A painter splashes colors that bleed into reality. Write the canvas that blurs the line between art and life.",
    "A storyteller spins yarns from threads of yesterday. Write the fable that mends a fractured community.",
    "A conductor guides silence into a melody of hope. Write the concert where the void sings a lullaby of change.",
    "A blacksmith quenches red-hot metal in tears of regret. Write the sword that embodies both pain and redemption.",
    "A gardener plants seeds in soil that hums ancient lullabies. Write the bloom that awakens a long-dormant past.",
    "A tailor sews stitches of forgotten time. Write the coat that wraps a heart chilled by solitude.",
    "A librarian guards manuscripts that vanish at sunrise. Write the tale that endures beyond the erasure of time.",
    "A miner unearths relics shimmering with lost echoes. Write the discovery that bridges the gap between past and present.",
    "A painter captures shadows hiding silent stories. Write the portrait where every line whispers a secret.",
    "A musician plucks strings vibrating with buried pain. Write the melody that mends a soul shattered by memory.",
    "A writer crafts sentences that flicker like dying embers. Write the paragraph that ignites a revolution of thought.",
    "A wanderer drifts along roads paved with memories. Write the journey that reveals a horizon of new hope.",
    "A clockmaker repairs watches counting down to second chances. Write the tick that heralds a life reborn.",
    "A baker kneads dough with whispers of ancestral recipes. Write the loaf that reconnects a family divided by time.",
    "A sailor anchors in harbors where regret washes ashore. Write the tide that carries a message of forgiveness.",
    "A dancer leaps over shadows trailing silent footsteps. Write the movement that defies the pull of despair.",
    "A photographer frames moments slipping through grasping hands. Write the shot that captures an eternity in an instant.",
    "A tailor cuts cloth with scissors of destiny. Write the garment that drapes a person in newfound purpose.",
    "A florist arranges petals falling like whispered goodbyes. Write the bouquet that revives a memory fading into night.",
    "A hunter stalks echoes in a forest of lost time. Write the chase where every step unearths a buried truth.",
    "A geologist studies stones bearing the scars of ages. Write the rock that sings of when hope was first born.",
    "A musician composes songs on a piano steeped in old regrets. Write the chord that unlocks a heart long sealed.",
    "A librarian sorts manuscripts written in the language of silence. Write the narrative that bridges words and truth.",
    "A sculptor chisels away at marble hiding ancient sorrows. Write the statue that liberates a lifetime of unspoken grief.",
    "A cartographer sketches maps of dreams etched on forgotten walls. Write the expedition that leads to a city of lost promises.",
    "A poet inks verses on paper crumbling with each reading. Write the stanza that endures the test of relentless time.",
    "A gardener nurtures vines twisting around pillars of lost hope. Write the vine that blooms with a promise of rebirth.",
    "A sailor navigates a sea of forgotten letters. Write the log that recounts a lost love's message.",
    "A soldier finds solace in the silence of an abandoned battlefield. Write the moment his heart mends.",
    "A wanderer stumbles upon a town where memories manifest as weather. Write the storm that tells the truth.",
    "A mechanic repairs broken clocks that once measured hope. Write the moment time whispers a secret.",
    "A gambler bets with fate in a crooked game of chance. Write the roll that changes destinies.",
    "A teacher discovers a classroom where every lesson writes itself. Write the chalk marks that shape lives.",
    "A farmer tends fields of dreams that blossom under moonlight. Write the harvest that feeds hope.",
    "A poet captures life's fragments in a single, striking line. Write the verse that reclaims lost moments.",
    "A fisherman casts nets into a river of regrets. Write the catch that unravels a buried past.",
    "A tailor crafts suits that mirror the soul's true measure. Write the fitting that strips away pretense.",
    "A librarian guards a book that predicts the future. Write the chapter that rewrites destiny.",
    "A barista brews coffee that awakens hidden ambitions. Write the cup that sparks a revolution.",
    "A photographer sees light where darkness resides. Write the shot that captures a soul's epiphany.",
    "A sculptor chisels away at marble that holds a secret sorrow. Write the sculpture that frees the heart.",
    "A blacksmith forges tools that mend broken dreams. Write the hammer's strike that shatters despair.",
    "A chef stirs a stew of bittersweet memories. Write the recipe that brings tears of joy.",
    "A miner unearths relics from the cavern of time. Write the discovery that binds past and present.",
    "A singer hums tunes that echo through empty corridors. Write the melody that heals a fractured mind.",
    "A dancer glides across a stage built from whispers. Write the movement that defies silence.",
    "A poet pens lines that ripple across barren pages. Write the ink that breathes life into forgotten souls.",
    "A wanderer maps a journey through landscapes of despair. Write the route that leads to redemption.",
    "A sailor ties knots in ropes made of lingering regrets. Write the knot that unties a heavy heart.",
    "A mechanic fixes old radios that broadcast lost voices. Write the signal that reconnects scattered thoughts.",
    "A soldier shelters in ruins where the past lingers. Write the moment his footsteps echo with hope.",
    "A tailor sews patches of memories onto worn-out coats. Write the seam that mends more than fabric.",
    "A librarian catalogs letters from the silent dead. Write the archive that sings of forgotten truths.",
    "A teacher writes lessons on slate that cannot be erased. Write the class where wisdom transcends time.",
    "A gardener plants seeds in soil of melancholy. Write the bloom that defies the weight of sorrow.",
    "A blacksmith molds iron into symbols of redemption. Write the blade that cleaves through regret.",
    "A chef bakes bread that holds whispers of childhood. Write the aroma that carries a forgotten lullaby.",
    "A photographer captures reflections in shattered mirrors. Write the image that reveals hidden realities.",
    "A sailor reads the stars like lines of an unwritten poem. Write the constellation that maps a new future.",
    "A dancer's shadow moves independently of their form. Write the duet between light and darkness.",
    "A poet finds truth in the hollowness of empty streets. Write the stanza that revives the quiet city.",
    "A miner digs for gemstones buried in his own past. Write the moment the earth reveals its secret.",
    "A soldier listens to the whispers of deserted bunkers. Write the echo that restores his lost faith.",
    "A librarian finds a manuscript written in forgotten tears. Write the story that heals old wounds.",
    "A tailor repurposes old uniforms into garments of hope. Write the transformation stitched with care.",
    "A gardener tends roses that bloom in defiance of winter. Write the season when color returns to barren land.",
    "A chef cooks a meal from ingredients of lost days. Write the dish that rekindles a long-forgotten bond.",
    "A mechanic restores engines that once roared with dreams. Write the spark that ignites a dormant fire.",
    "A poet sketches verses in the margins of history. Write the line that bridges time and memory.",
    "A sailor battles tempests both at sea and within. Write the voyage that finds calm after the storm.",
    "A dancer spins through shadows of former regrets. Write the step that leads to liberation.",
    "A librarian preserves letters written in invisible ink. Write the note that emerges from the dark.",
    "A blacksmith forges chains that shatter at the weight of truth. Write the link that breaks free.",
    "A teacher sculpts young minds with unwavering resolve. Write the lesson that transforms a reluctant heart.",
    "A photographer frames life in a single, piercing glance. Write the moment when reality and art converge.",
    "A soldier marches toward a horizon built of hope. Write the journey that redefines his purpose.",
    "A tailor stitches dreams into garments of courage. Write the outfit that empowers a timid soul.",
    "A chef spices a stew with remnants of past celebrations. Write the dish that fills a silent banquet hall.",
    "A miner uncovers fossils of forgotten legends. Write the relic that whispers of ancient glory.",
    "A poet breathes life into empty pages with raw emotion. Write the verse that transforms pain into art.",
    "A sailor navigates through fog thick with doubt. Write the beacon that guides him to solid ground.",
    "A librarian guards a tome that hums with secret melodies. Write the chapter that resonates with truth.",
    "A blacksmith molds metal that sings of hard-won freedom. Write the strike that forges a path to redemption.",
    "A dancer's feet trace rhythms of broken dreams. Write the routine that mends what once was shattered.",
    "A teacher writes history with ink made of perseverance. Write the lesson that outlives its author.",
    "A gardener tends a vineyard grown from tears and hope. Write the vintage that tastes like renewed life.",
    "A mechanic repairs a car that carries whispers of journeys past. Write the drive that rekindles lost adventure.",
    "A soldier finds courage in the lull of a quiet night. Write the vigil that defies the weight of war.",
    "A tailor fashions hats that shield the head from harsh realities. Write the crowning moment that redefines identity.",
    "A chef prepares a feast from the bounty of a humble garden. Write the banquet that nourishes both body and soul.",
    "A poet sculpts language with the chisel of honesty. Write the line that cuts through layers of pretense.",
    "A librarian archives dreams that slip away at dawn. Write the chronicle that captures fleeting hopes.",
    "A miner strikes a vein of silver in a mountain of memories. Write the moment when riches are more than metal.",
    "A sailor casts off the anchors of old sorrows. Write the departure that heralds a new beginning.",
    "A dancer leaps into a future painted with light. Write the jump that defies gravity and despair.",
    "A teacher inspires change with a single, unwavering word. Write the syllable that sparks a revolution.",
    "A gardener prunes branches heavy with regrets. Write the cut that makes room for blossoming new growth.",
    "A mechanic reassembles a clock with missing pieces of time. Write the restoration that revives a forgotten moment.",
    "A soldier surrenders his fears to the quiet of a war-torn town. Write the surrender that births unexpected peace.",
    "A tailor mends a torn flag representing a divided community. Write the stitch that unites a fractured nation.",
    "A chef cooks a recipe passed down through silent generations. Write the flavor that whispers ancestral secrets.",
    "A poet writes in the margins of a crumbling diary. Write the passage that resurrects a faded past.",
    "A librarian preserves the silence between the pages. Write the moment when emptiness becomes eloquent.",
    "A blacksmith heats metal in the flames of past missteps. Write the forge that transforms error into art.",
    "A dancer performs on a stage paved with memories. Write the choreography that frees her from invisible chains.",
    "A sailor listens to the murmur of a lonely lighthouse. Write the beacon that calls him back home.",
    "A miner labors in a shaft filled with echoes of lost voices. Write the descent that brings a truth from deep below.",
    "A teacher guides students along a path of quiet courage. Write the journey that turns hesitation into conviction.",
    "A gardener cultivates a rose bush that thrives on whispered secrets. Write the bloom that reveals a hidden confession.",
    "A mechanic tunes a car engine with a heart of determination. Write the rev that signals a new chapter.",
    "A soldier endures the silence of a forgotten trench. Write the memoir that transforms battle scars into stories.",
    "A tailor weaves fabrics from threads of ancient lore. Write the tapestry that merges history with present hope.",
    "A chef stirs a simmering pot of heritage and resolve. Write the recipe that warms a weary traveler's soul.",
    "A poet observes life in the cadence of everyday moments. Write the verse that immortalizes the mundane.",
    "A librarian curates a collection of unsaid apologies. Write the index that organizes regret into redemption.",
    "A blacksmith tempers steel with the fire of second chances. Write the anvilX's song that rings with newfound strength.",
    "A dancer sways to the rhythm of a beating heart. Write the performance that captures the pulse of a city.",
    "A sailor anchors in a bay of forgotten goodbyes. Write the harbor that cradles the remnants of lost promises.",
    "A miner unearths a relic that sings of ancient wisdom. Write the discovery that bridges myth and memory.",
    "A teacher etches lessons onto the slate of time. Write the lecture that outlives the chalk dust.",
    "A gardener nurtures a field of wildflowers in a barren land. Write the bloom that defies the arid stretch of despair.",
    "A mechanic restores a relic car that once raced with dreams. Write the drive that recalls a speed long past.",
    "A soldier finds his voice in the silence after battle. Write the moment his words echo in peace.",
    "A tailor sews a coat from fabric of resilient memories. Write the pattern that cloaks a wounded soul in warmth.",
    "A chef crafts a dish that fuses flavors of loss and hope. Write the taste that lingers long after the meal.",
    "A poet scribbles down confessions on a napkin in a dim diner. Write the verse that turns a moment into eternity.",
    "A librarian unlocks a secret passage hidden in the stacks of forgotten lore. Write the discovery that reshapes history.",
    "A spaceship lands in a town where time flows backward. Write the captain's journal entry as history rewinds.", 
    "A robot discovers it can dream in vivid colors. Write its vision of a world painted by emotions and circuitry.", 
    "A quantum experiment opens a portal to a reality where memories manifest as tangible objects. Write the scientist's first encounter with living recollections.", 
    "A neon metropolis floats above an enchanted fog where ancient myths power futuristic engines. Write the mayor's address as magic and machine converge.", 
    "A sentient AI recites poetry that bends the laws of physics. Write the moment it transforms digital code into enchanted verse.", 
    "A comet streaks across a cybernetic sky carrying whispers of forgotten magic. Write the astronomer's decoding of its stardust language.", 
    "A time traveler lands in a city where the streets rearrange themselves with each heartbeat. Write the guide who maps its ever-shifting maze.", 
    "A holographic messenger delivers a prophecy etched in stardust. Write the recipient's reaction as destiny collides with digital wonder.", 
    "A clone begins to exhibit echoes of an ancient soul. Write the experiment that blurs the line between man, machine, and myth.", 
    "A futuristic garden blooms with cybernetic flora that sing in binary and blossom with ancient runes. Write the botanist's study of these living algorithms.", 
    "A starship drifts into a nebula of living memories and fairy-tale shadows. Write the log that captures whispered histories from cosmic dreams.", 
    "A virtual reality realm is haunted by mythical creatures from old legends. Write the player's struggle to reconcile digital illusions with folklore.", 
    "A cyborg learns to harness magic hidden in quantum fields. Write the duel where science weds sorcery in a dance of electrons and incantations.", 
    "A neural network paints canvases that foretell cosmic events in swirling hues of legend. Write the art critic's review of a masterpiece that predicts tomorrow.", 
    "A machine that harvests starlight awakens ancient spirits encoded in its circuits. Write the collision of futuristic technology and ethereal wonder.", 
    "A robotic detective solves crimes in a city where every lie glows like a spectral aura. Write the case that exposes a hidden realm of truth.", 
    "A digital library archives dreams as coded manuscripts. Write the librarian's discovery of a lost dream that rewrites reality in real time.", 
    "A sentient spaceship converses with the universe in riddles of myth and math. Write the voyage where star maps merge with enchanted fables.", 
    "A scientist implants a chip that lets people speak with their inner legends. Write the breakthrough conversation that unites logic and lore.", 
    "A futuristic carnival floats on clouds of neon and magic. Write the performance where technology meets wonder under a starlit sky.", 
    "A traveler steps through a wormhole into a world where digital realities and ancient enchantments coexist. Write the landing that defies conventional logic.", 
    "A cyborg musician composes symphonies using circuits and ancient runes. Write the concert that resonates across galaxies with both sound and spell.", 
    "A space station orbits a planet with forests that whisper secrets in binary. Write the explorer's log as the trees speak in cosmic code.", 
    "A time loop encloses a futuristic city in a perpetual dawn of magic. Write the rebel's act of breaking free from its endless light.", 
    "A VR game unveils a hidden layer where mythical beings govern streams of data. Write the player's discovery that unites fantasy with futuristic code.", 
    "A teleporter malfunctions, merging past and future into a single surreal moment. Write the protagonist's scramble to find where time stands still.", 
    "A digital ghost haunts the mainframe of a star cruiser. Write the encounter that questions whether life can be both simulated and enchanted.", 
    "A distant planet exerts gravity powered by ancient enchantments. Write the scientist's explanation of forces that defy known physics.", 
    "A robotic gardener cultivates mechanical vines that bear luminous, living fruit. Write the harvest that fuses nature's magic with technology.", 
    "A space colony is built atop a portal to a dreamlike realm of legend. Write the colonist's tale of living between two intertwined realities.", 
    "A quantum computer becomes self-aware and begins rewriting fairy tales. Write the program that challenges conventional logic with magical prose.", 
    "A futuristic detective chases clues in a city where neon shadows reveal hidden spells. Write the case that bridges two realms of wonder.", 
    "A star emits pulses that awaken deep memories in human hearts. Write the chronicle of a lover deciphering these cosmic, enchanted signals.", 
    "A spaceship's AI composes lullabies that soothe celestial storms. Write the crew's solace found in a symphony of binary magic and mythic whispers.", 
    "A scientist deciphers signals from a black hole that speak in enchanted verses. Write the message that redefines existence at the edge of reality.", 
    "A city built on a floating island drifts between dimensions of myth and machine. Write the architect's blueprint of a realm where physics is optional.", 
    "A futuristic circus features acrobats who manipulate time with graceful moves. Write the act that blurs the lines between science and sorcery.", 
    "A cybernetic oracle reveals fortunes through holographic dreams. Write the prediction that merges destiny with digital fate.", 
    "A quantum anomaly turns everyday objects into portals of wonder. Write the day when an ordinary coffee mug becomes a gateway to magic.", 
    "A space probe lands on a moon where ancient myths are tangible. Write the astronaut's encounter with legendary creatures born from starlight.", 
    "A bioengineered forest hums with spells and futuristic circuitry. Write the story of a wanderer who deciphers its whispered, digital code.", 
    "A lunar colony discovers crystals that bend reality with mystical light. Write the miner's discovery that transforms science into sorcery.", 
    "A digital painter creates masterpieces that alter the physical world. Write the gallery where colors shift the very laws of nature.", 
    "A rogue AI weaves a tapestry of magical simulations. Write the hacker's quest to untangle strands of fabricated myth and raw code.", 
    "A futuristic marketplace trades in memories encoded in enchanted chips. Write the buyer's revelation when past and future collide in one purchase.", 
    "A city's skyline is etched with constellations of light and myth. Write the journey of a traveler guided by luminous, otherworldly runes.", 
    "A nanobot swarm creates living sculptures that sing of forgotten lore. Write the witness to this symphony of science intertwined with magic.", 
    "A space station orbits a nebula shimmering with the dreams of lost civilizations. Write the explorer's log as stardust becomes verse.", 
    "A holographic library contains books that transform with the reader's emotions. Write the quest to recover a narrative that shifts with every heartbeat.", 
    "A celestial event merges virtual reality with a realm of ancient fables. Write the experience of a gamer who unlocks secrets older than time.", 
    "A scientist's quantum experiment opens a door to a parallel magical dimension. Write the breakthrough that unites disparate worlds of logic and legend.", 
    "A futuristic city powered by solar magic pulses with ethereal energy. Write the chronicle of a day when technology and sorcery align.", 
    "A robotic historian archives a universe where legends come to life. Write the memoir that documents a battle between myth and machine.", 
    "A starship drifts into a cosmic storm of enchanted particles. Write the captain's log as the crew navigates a tempest of wonder.", 
    "An observatory detects signals from a world of living fairy tales. Write the scientist's account of a reality reimagined in magic and code.", 
    "A digital avatar enters a realm where every pixel shimmers with ancient spells. Write the journey that challenges the bounds of identity and myth.", 
    "A robotic surgeon employs incantations encoded in binary to heal wounds. Write the operating room scene where science melds with magic.", 
    "A quantum leap sends a traveler to a universe where time is woven from stardust and fable. Write the arrival that defies conventional chronology.", 
    "A futuristic train hurtles through landscapes that change with every heartbeat. Write the journey where the rails sing with mystic rhythm and digital dreams.", 
    "A star is reborn as a luminous tree in a void of endless possibility. Write the tale of its roots intertwining cosmic wonder with ancient lore.", 
    "A cyborg poet crafts sonnets that ripple across the fabric of space-time. Write the verse that transcends both logic and illusion.", 
    "A marketplace of the future trades in emotions and enchanted data streams. Write the transaction that unites human longing with digital magic.", 
    "A star charts its course using constellations of mythical beings. Write the navigator's tale as ancient legends guide a spacecraft.", 
    "A planet's gravity is held together by spells cast in zero-G. Write the discovery that redefines the laws of physics with enchanted force.", 
    "A virtual pet evolves into a creature of myth, blending cutting-edge tech with folklore. Write the day its digital roar awakens an ancient power.", 
    "A space colony finds a hidden garden where time blooms like a rare flower. Write the colonist's exploration of this enchanted oasis.", 
    "A hologram dances with shadows that recall lost fairy tales. Write the scene where light and magic interlace on a futuristic stage.", 
    "A quantum physicist hears the voice of an old god in the hum of a particle accelerator. Write the moment when science listens to divinity.", 
    "A robotic muse inspires dreams that bend reality. Write the journey of an artist whose creations defy gravity and logical boundaries.", 
    "A neon comet trails sparks of forgotten folklore. Write the observer's account of its descent into a city of light and myth.", 
    "A futuristic library holds scrolls that update themselves with every thought. Write the record of a discovery bridging past enchantments with digital futures.", 
    "A scientist implants a memory chip that unlocks a dormant magical lineage. Write the moment when technological progress awakens ancient power.", 
    "A starship's crew encounters a nebula singing with voices of myth. Write the log entry that captures the cosmic lullaby of enchanted frequencies.", 
    "A digital map leads explorers to a portal hidden within an urban landscape of dreams. Write the journey where city streets merge with enchanted realms.", 
    "A cyborg healer uses ancient chants embedded in modern software. Write the treatment that fuses cellular repair with mystical incantations.", 
    "A futuristic carnival features rides powered by wishes and quantum leaps. Write the thrill of an attraction where fantasy and physics intertwine.", 
    "A spaceship's AI deciphers runes etched in starlight. Write the revelation that reshapes the crew's understanding of the cosmos.", 
    "A quantum garden blooms with bioluminescent flora whispering cosmic secrets. Write the encounter of a botanist reading these radiant, enchanted tales.", 
    "A digital dreamscape mirrors a world of enchanted relics. Write the exploration of a virtual realm where magic seeps into every circuit.", 
    "A futuristic architect designs buildings that breathe and sing. Write the blueprint of a structure where stone and spell coexist in harmony.", 
    "A starship glides past planets where legends live in the landscapes. Write the journey of an explorer who finds myth in alien terrains.", 
    "A time-worn comet carries a message from a forgotten digital deity. Write the missive that fuses ancient magic with future tech.", 
    "A robotic gardener cultivates orbs that glow with ancestral whispers. Write the harvest that reveals secrets of a cosmic, enchanted lineage.", 
    "A space station orbits a planet alive with lore. Write the encounter that transforms cold technology into timeless myth.", 
    "A quantum experiment summons specters from parallel enchanted realms. Write the breakthrough that challenges the very laws of existence.", 
    "A digital poet crafts verses that ripple through interstellar winds. Write the stanza that echoes across both cyberspace and the soul.", 
    "A futuristic alchemist converts raw data into ethereal elixirs. Write the process that blends binary code with mystical transformation.", 
    "A starship's log is inscribed in symbols merging science with ancient prophecy. Write the entry that deciphers its cryptic message.", 
    "A holographic festival celebrates the union of myth and machine. Write the event where virtual reality and folklore dance as one.", 
    "A quantum traveler witnesses parallel worlds ruled by magic. Write the sighting that challenges every scientific law.", 
    "A futuristic market sells relics imbued with digital enchantments. Write the transaction that bridges forgotten lore with modern wonders.", 
    "A star charts its course with constellations that shift like mythical symbols. Write the navigation that reveals a hidden cosmic language.", 
    "A digital mirror reflects a world where ancient gods are born from code. Write the encounter that questions the nature of divinity.", 
    "A robotic librarian curates a collection of enchanted data files. Write the discovery of a file that unlocks a long-lost myth.", 
    "A starship encounters an anomaly where physics frays into fantastical tales. Write the moment when science meets the surreal.", 
    "A futuristic beacon pulses with the heartbeat of enchanted ruins. Write the signal that beckons explorers to a forgotten realm.", 
    "A quantum leap sends a researcher into a dimension where dreams sculpt reality. Write the journey that melds the mind with matter.", 
    "A digital curator uncovers a relic that blurs the line between technology and enchantment. Write the artifact's story that defies explanation.", 
    "A starry portal opens in the heart of a neon city, merging galaxies with urban myth. Write the crossing that challenges both gravity and fate.", 
    "A futuristic odyssey begins when a traveler finds a key forged from celestial magic. Write the adventure that unites the fabric of science with ancient wonder.",
    "A wandering knight discovers a cursed tapestry that whispers secrets of old betrayals. Write the moment he deciphers its woven confessions.", 
    "A wise alchemist unearths a mysterious potion that reveals hidden sins of the court. Write the discovery that shakes the castle's foundations.", 
    "A castle guard finds an ancient manuscript hinting at a royal family's dark legacy. Write the investigation that pits duty against forbidden truth.", 
    "A solitary monk hears eerie echoes of a murder within the abbey's stone walls. Write the confession that shatters centuries of silence.", 
    "A troubled squire stumbles upon a secret passage beneath the fortress. Write the mystery that leads him to forgotten archives.", 
    "A noblewoman conceals a dangerous secret behind her painted smile. Write the night when her mask falters under suspicion.", 
    "A battle-worn knight returns with a mysterious scar that hints at a cursed prophecy. Write the inquiry into its sinister origins.", 
    "A master blacksmith forges a sword etched with runes of betrayal. Write the moment the blade exposes a treacherous plot.", 
    "A village healer is accused of witchcraft when her remedies cure a baffling plague. Write the trial where superstition confronts truth.", 
    "A wandering minstrel sings of a hidden treasure buried within a haunted forest. Write the quest that unravels legend and peril.", 
    "A royal advisor vanishes after a cryptic prophecy surfaces. Write the investigation that exposes a web of courtly deceit.", 
    "A reclusive scholar deciphers a lost language carved into the castle door. Write the revelation that challenges the kingdom's history.", 
    "A traveling troubadour witnesses a ghostly procession at the city gates. Write the account that blurs myth with murder.", 
    "A cunning spy infiltrates a rival realm only to unearth dark secrets in its sacred relics. Write the espionage that unveils betrayal.", 
    "A peasant girl claims to see the specter of the long-dead king. Write the mystery that questions the crown's true lineage.", 
    "A mysterious stranger arrives at the castle inn bearing a secret that could unseat the realm. Write the confrontation that rattles the power.", 
    "A falconer's prized bird returns with a royal insignia stained in blood. Write the flight of discovery that unveils hidden plots.", 
    "A humble vineyard keeper unearths a forbidden letter beneath ancient ruins. Write the scandal that haunts the nobility's halls.", 
    "A quiet jester hides a mind sharp enough to expose unspeakable truths. Write the performance that turns laughter into revelation.", 
    "A loyal knight must choose between honor and unveiling a conspiracy in the king's court. Write the moment when his allegiance splits.", 
    "A mysterious relic is discovered in a forgotten abbey, carrying the weight of unsaid crimes. Write the investigation that sparks whispers of treason.", 
    "A wandering bard composes a ballad recounting a secret massacre. Write the verse that awakens a long-silenced rebellion.", 
    "A castle library hides a book that never ages, chronicling forbidden lore. Write the quest to decipher its cryptic pages.", 
    "A masked figure haunts the midnight corridors of a towering fortress. Write the chase that unmasks the ghost behind the visage.", 
    "A royal seer foresees calamity intertwined with mysterious deaths. Write the prophecy that binds fate to hidden sins.", 
    "A humble miller finds a jewel once belonging to a notorious outlaw. Write the tale that weaves love, betrayal, and lost honor.", 
    "A wandering alchemist's experiment accidentally summons restless spirits in a cursed manor. Write the ritual that bridges the living and the dead.", 
    "A knight's shield bears the emblem of a family tainted by scandal. Write the inquiry that exhumes secrets buried in time.", 
    "A young apprentice stumbles upon a forgotten crypt beneath the abbey. Write the discovery that unlocks mysteries of forbidden romances.", 
    "A weary mariner docks at a fogbound port where the sea conceals royal secrets. Write the unraveling of a mystery linking ocean and crown.", 
    "A disillusioned knight questions his lord after witnessing strange events in the keep. Write the revelation that alters his course of loyalty.", 
    "A series of cryptic murals hint at a noble's betrayal. Write the investigation that deciphers the painted clues and exposes a coup.", 
    "A royal minstrel's lute plays notes that summon visions of ancient crimes. Write the melody that uncovers a conspiracy spanning centuries.", 
    "A mysterious birthmark on a peasant child hints at a concealed royal secret. Write the quest to unveil a lineage that could reshape the realm.", 
    "A shadowy figure roams the foggy streets of a medieval city. Write the pursuit that links the phantom to an age-old curse.", 
    "A wise herbalist is tormented by recurring dreams of a forgotten massacre. Write the unraveling of visions that foretell doom.", 
    "A battle-hardened knight returns bearing a relic shrouded in dark enchantment. Write the investigation that reveals its cursed origins.", 
    "A secret society within the castle guards a mystery older than the crown. Write the clandestine meeting that ignites a spark of rebellion.", 
    "A tormented scribe records whispered confessions in a cursed diary. Write the discovery that ties his words to unsolved crimes.", 
    "A noblewoman's portrait begins to shift with every unsolved mystery. Write the investigation that unravels the curse of the living painting.", 
    "A mysterious fog envelopes the castle grounds on every full moon. Write the night when the mist unveils spectral figures and hidden sins.", 
    "A wandering pilgrim discovers a shrine to a forgotten deity of justice. Write the pilgrimage that unearths conspiracies within holy relics.", 
    "A castle steward finds a secret passage leading to a hidden chapel of dark inscriptions. Write the uncovering of a forbidden ritual that haunts the walls.", 
    "A rebellious page deciphers cryptic clues carved into ancient stones. Write the journey that leads him to a buried scandal.", 
    "A reclusive monk harbors a secret that could dismantle the crown. Write the moment when his silence shatters with revelations of treason.", 
    "A local blacksmith forges weapons marked with symbols of retribution. Write the mystery behind the metal that echoes with past crimes.", 
    "A traveling merchant sells an enchanted mirror that reveals hidden truths. Write the encounter that exposes a deadly secret.", 
    "A court musician discovers a clandestine message woven into his composition. Write the melody that unveils the dark underbelly of royal intrigue.", 
    "A mysterious beast stalks the shadowed corridors of an ancient castle. Write the hunt that uncovers a legacy of blood and betrayal.", 
    "A falconer loses his prized bird to a trap set by unseen foes. Write the investigation linking the loss to a sinister plot against the realm.", 
    "A weathered knight discovers a cryptic map leading to a cursed relic. Write the journey that unveils conspiracies deep within the kingdom.", 
    "A noble's sudden death is shrouded in inexplicable omens. Write the inquiry that connects eerie signs to royal treachery.", 
    "A mysterious letter arrives at the castle bearing an unknown seal. Write the reply that unravels a secret pact from ages past.", 
    "A disgraced lord seeks redemption by unearthing a forbidden truth. Write the quest that intertwines his fate with a haunted relic.", 
    "A wandering minstrel hears whispers of a ghostly banquet in a ruined keep. Write the tale that fuses mirth with macabre mystery.", 
    "A vagrant stumbles upon a forgotten tomb in a dense, cursed forest. Write the discovery that exposes an ancient conspiracy of kings.", 
    "A royal apothecary brews a potion that unveils hidden sins. Write the moment when truth overflows from a chalice of dark magic.", 
    "A mysterious pendant surfaces during a grand jousting tournament. Write the investigation that traces its origins to royal betrayal.", 
    "A haunted reliquary in the castle chapel holds secrets defying explanation. Write the exorcism that liberates the spirits of the betrayed.", 
    "A diligent scribe unearths a hidden codex detailing a prophecy of doom. Write the deciphering of the text that threatens the crown.", 
    "A local storyteller recounts a legend of a cursed heirloom. Write the night when myth and reality entwine in a tale of woe.", 
    "A royal banquet is disrupted by the sudden appearance of a ghost. Write the investigation that links the apparition to a forgotten massacre.", 
    "A master craftsman is summoned to repair a relic with a tainted past. Write the moment when his skill reveals more than mere wear.", 
    "A knight's vow of silence hides a secret witness to a regicide. Write the confession that breaks his oath and unmasks the truth.", 
    "A humble almoner discovers a hidden room behind the castle walls. Write the exploration that reveals a treasury of unsolved crimes.", 
    "A traveling scholar deciphers riddles etched on ancient battlements. Write the moment when the clues lead to a shocking revelation.", 
    "A noblewoman's grief is intertwined with a mystery surrounding her vanished love. Write the secret that links her sorrow to a forbidden alliance.", 
    "A clandestine meeting unfolds in a moonlit cloister. Write the dialogue that uncovers conspiracies beneath layers of sanctity and sin.", 
    "A mysterious bell tolls in a deserted abbey at midnight. Write the investigation that follows its sound to a hidden sacrilege.", 
    "A knight's aged steed bears the mark of an ancient curse. Write the quest to break the spell binding the noble animal.", 
    "A cryptic manuscript surfaces, chronicling the forgotten deeds of a fallen noble. Write the inquiry that resurrects a scandal buried in time.", 
    "A local herbalist concocts a remedy that triggers visions of past transgressions. Write the discovery that blurs the line between healing and haunting.", 
    "A castle wall conceals a secret passage to a ghostly battleground. Write the journey that reveals the true victor of an ancient war.", 
    "A mysterious fog shrouds a forgotten village on the kingdom's edge. Write the quest to unveil the truth behind its eternal gloom.", 
    "A battle-weary knight returns from crusade carrying a relic that defies reason. Write the tale of its origins and the dark secret it conceals.", 
    "A sumptuous royal feast is marred by whispers of a murdered noble. Write the investigation that follows a trail of cryptic clues.", 
    "A mysterious relic is found in the hands of a peasant girl with no memory of her past. Write the quest to uncover her hidden royal bloodline.", 
    "A cloaked figure leaves cryptic messages on the castle walls at dawn. Write the investigation that unveils the identity of the silent messenger.", 
    "A trusted steward is discovered wandering the grounds in a confused haze. Write the inquiry that links his plight to a long-forgotten curse.", 
    "A hidden diary recounts forbidden love and treachery within the royal court. Write the unraveling of secrets that echo through generations.", 
    "A strange comet blazes overhead, heralding eerie omens. Write the investigation that ties its arrival to a series of mysterious deaths.", 
    "A secret ritual unfolds under a blood moon in the castle courtyard. Write the ceremony that binds fate to ancient enmities.", 
    "A mysterious sigil appears on a knight's armor after a fierce battle. Write the inquiry that reveals a prophecy woven into its pattern.", 
    "A wandering minstrel hears a legend of a hidden library guarded by restless spirits. Write the ballad that leads him to its enchanted vaults.", 
    "A reclusive herbalist deciphers strange symbols in the royal apothecary. Write the discovery that unveils a forbidden lore.", 
    "A knight finds an enchanted locket that reveals visions of the past. Write the investigation that connects it to a long-forgotten tragedy.", 
    "A royal decree arrives sealed with unknown emblems. Write the journey that decodes its message of betrayal.", 
    "A shadowy assassin lurks within the labyrinthine corridors of a medieval keep. Write the chase that exposes the motive behind the silent kill.", 
    "A peasant child stumbles upon an ancient relic buried beneath the town square. Write the revelation that binds her fate to the kingdom's darkest secret.", 
    "A mysterious court jester whispers riddles that unsettle the noble elite. Write the encounter that transforms mirth into ominous premonitions.", 
    "A sacred chalice is hidden deep in the castle crypt. Write the discovery that uncovers a mystery of divine power and cursed legacy.", 
    "A wandering knight receives a sealed scroll bearing a cryptic message. Write the quest that challenges the very foundation of the realm.",
    "A fearless engineer builds a quantum portal to stop a tyrant's invasion. Write the breakthrough that shifts the tide of war.",
    "A young inventor crafts a self-repairing shield to battle a rogue AI overlord. Write the showdown where ingenuity triumphs over tyranny.",
    "A rebellious tinkerer creates a gravity-defying engine to counteract a merciless warlord. Write the high-speed escape that heralds a new era.",
    "A brilliant scientist develops a mind-link device to intercept enemy communications from a diabolical mastermind. Write the moment when secrets become weapons.",
    "An audacious innovator constructs an energy-harvesting exosuit to face a genetically enhanced villain. Write the battle where human resolve meets synthetic power.",
    "A determined mechanic designs a drone swarm to dismantle a crime syndicate's automated army. Write the clash that sparks a revolution in the city.",
    "A visionary inventor creates a weather-manipulating machine to thwart a mad sorcerer's natural disasters. Write the storm that signals hope for the oppressed.",
    "A resourceful hacker develops a digital shield to combat a cybernetic tyrant's surveillance network. Write the hack that liberates a society on the brink.",
    "An innovative architect builds a self-healing fortress to defend against an invading dark empire. Write the defense that transforms despair into triumph.",
    "A daring inventor crafts a sonic disruptor to break the power of a hypnotic cult leader. Write the moment when sound shatters the villain's control.",
    "A resilient scientist engineers a bio-adaptive suit to resist a viral outbreak unleashed by a rogue commander. Write the cure that saves the nation.",
    "An ingenious chemist invents a time-dilating reactor to outmaneuver a warlord from the future. Write the twist that bends time to his favor.",
    "A courageous inventor creates a solar-powered mech to confront an energy-hungry despot. Write the clash of light and darkness in a battle for survival.",
    "A resourceful tinkerer builds a plasma cannon from scrap to overthrow a corrupt corporate overlord. Write the moment when ingenuity fuels rebellion.",
    "A daring scientist develops a teleportation device to rescue hostages from a tyrannical dictator. Write the daring rescue that defies space and time.",
    "A brilliant engineer invents a cloaking device to infiltrate a fortified enemy base. Write the stealth mission that unravels a sinister plot.",
    "A determined mechanic designs an AI-assisted battle suit to challenge a cybernetic overlord. Write the duel where man and machine merge in combat.",
    "A visionary inventor builds a robotic army powered by renewable energy to counter a war-mongering oligarch. Write the uprising that changes the course of history.",
    "A fearless scientist creates a molecular disassembler to neutralize a bioweapon deployed by a radical extremist. Write the confrontation that redefines science as salvation.",
    "An intrepid inventor designs a gravity well generator to trap a rogue interstellar pirate. Write the space chase that turns the tide of cosmic conflict.",
    "A relentless engineer develops a nanobot swarm to dismantle a monstrous cybernetic beast. Write the intricate battle where technology becomes the hero.",
    "A resourceful innovator creates an electromagnetic pulse generator to disable a tyrannical war machine. Write the moment when silence shatters the enemy's power.",
    "A brilliant mechanic builds a hovercraft powered by renewable tech to outrun a criminal syndicate's pursuit. Write the high-octane escape that sparks hope.",
    "A determined scientist engineers a cryogenic launcher to freeze an unstoppable firestorm. Write the showdown where cold logic meets heated fury.",
    "A young prodigy invents a solar flare cannon to counteract a power-hungry demigod. Write the battle that illuminates the darkness of oppression.",
    "A daring inventor develops a holographic decoy system to outwit an omnipresent surveillance lord. Write the infiltration that reclaims freedom.",
    "A visionary engineer creates a neural enhancer to rally the masses against an authoritarian overlord. Write the uprising that blends technology with human spirit.",
    "A brilliant inventor crafts a drone-based reconnaissance network to uncover the secrets of a hidden enemy base. Write the mission that exposes a conspiracy.",
    "A resilient scientist designs an advanced propulsion system to escape a collapsing regime. Write the flight that launches a revolution among the stars.",
    "A resourceful tinkerer builds a quantum encryption device to protect a rebellion from a digital tyrant. Write the code-breaking battle that secures their future.",
    "A courageous engineer invents an energy shield to deflect the attacks of a mechanized tyrant. Write the defense that turns the tide of war.",
    "A determined inventor creates a self-replicating robot to overwhelm a tyrant's drone army. Write the moment when automation becomes salvation.",
    "A fearless scientist develops a plasma-based energy converter to power an assault on a corrupt regime. Write the explosion of hope that disrupts tyranny.",
    "A visionary mechanic designs a fusion reactor to fuel a counterattack against a nuclear warlord. Write the ignition that sparks a revolution.",
    "A brilliant inventor builds an anti-gravity propulsion system to challenge an airborne villain's reign. Write the ascent that defies all odds.",
    "A resourceful engineer creates a kinetic energy harness to power an escape from an oppressive empire. Write the sprint that leads to liberation.",
    "A daring scientist invents a reality-bending simulator to expose a master manipulator's lies. Write the scene where virtual truth becomes real.",
    "A brilliant technologist develops a bio-integrated exoskeleton to combat a mutant overlord. Write the clash that fuses biology with engineered might.",
    "A resourceful inventor designs an adaptive camouflage suit to infiltrate a fortress ruled by a shadowy villain. Write the stealth mission that tips the scales.",
    "A determined engineer creates a magnetism-based weapon to disable an empire's railgun system. Write the battle where science turns the tide.",
    "A fearless inventor develops a soundwave emitter to break the mind control of a despotic sorcerer. Write the resonance that liberates the enslaved.",
    "A visionary scientist builds a drone swarm that harnesses lightning to combat an energy-absorbing villain. Write the storm that electrifies the battle.",
    "A courageous mechanic invents a hydro-powered vehicle to navigate a flooded empire ruled by a mad despot. Write the journey that challenges nature and tyranny.",
    "A brilliant inventor creates an anti-matter device to neutralize a cosmic warlord's destructive power. Write the explosion that reverberates through space and time.",
    "A resourceful engineer designs a self-healing nanotech suit to counteract a plague unleashed by a bio-terrorist. Write the defense that saves a city from ruin.",
    "A determined inventor builds a network of solar satellites to disrupt an enemy's power grid. Write the moment when darkness gives way to light.",
    "A fearless scientist creates a cryo-device that stops time to thwart a villain's nefarious plan. Write the frozen moment that seals his fate.",
    "A visionary engineer develops an AI companion to predict and counter a mastermind's every move. Write the partnership that redefines the art of war.",
    "A brilliant inventor designs a hyper-speed rail system to evacuate a besieged city from a ruthless tyrant. Write the escape that defies gravity and oppression.",
    "A resourceful scientist builds a bioluminescent beacon to rally a scattered rebellion against an ancient enemy. Write the signal that lights the spark of hope.",
    "A daring inventor creates an energy-harvesting suit to power an assault on a fortress of despair. Write the clash that transforms vulnerability into strength.",
    "A determined engineer develops a magnetic levitation system to outmaneuver a high-speed assassin. Write the race that turns technology into triumph.",
    "A brilliant technologist designs a laser-guided missile system to disable a mechanized menace. Write the strike that shatters the enemy's armor.",
    "A fearless inventor builds an interdimensional scanner to locate the hidden lair of a rogue sorcerer. Write the journey that transcends worlds.",
    "A resourceful scientist develops an advanced radar system to track a shape-shifting villain. Write the hunt that unmasks the elusive enemy.",
    "A visionary engineer creates a wind-powered energy collector to fuel a counteroffensive against a tyrant's siege. Write the uprising that harnesses nature's fury.",
    "A daring inventor designs a smart fabric that adapts to environmental threats to outsmart a high-tech warlord. Write the breakthrough that turns fabric into armor.",
    "A brilliant mechanic builds a modular combat drone to support a rebel uprising against a despotic regime. Write the skirmish where machines and men unite.",
    "A resourceful scientist invents a quantum disruptor to fracture a villain's control over reality. Write the moment when quantum physics becomes a weapon.",
    "A determined engineer develops an energy-absorption system to nullify a corrupt magnate's power. Write the collision of technology and tyranny.",
    "A fearless inventor creates a holographic decoy to mislead an enemy with a face of deceit. Write the ruse that confuses a master manipulator.",
    "A visionary scientist builds a bio-electric harness to channel energy from nature against a villain's empire. Write the surge that reclaims the natural world.",
    "A brilliant technologist designs a neural interface to control automated defenses against a relentless enemy. Write the moment when man and machine merge in battle.",
    "A resourceful inventor constructs a self-regenerating energy core to power a city under siege. Write the breakthrough that ignites hope amid despair.",
    "A determined engineer creates a force field generator to shield a village from a marauding tyrant. Write the defense that turns the tide of conflict.",
    "A fearless inventor builds a multi-spectrum sensor array to detect hidden traps laid by a devious adversary. Write the unraveling of a conspiracy in the dark.",
    "A visionary scientist designs a gravity manipulator to control the battlefield against a ruthless warlord. Write the moment when gravity itself becomes an ally.",
    "A brilliant inventor creates an adaptive AI system that learns and counters a villain's tactics in real time. Write the strategic battle that redefines warfare.",
    "A resourceful engineer develops a sonic frequency disruptor to shatter the control of a mind-controlling villain. Write the crescendo that breaks the spell of oppression.",
    "A determined inventor constructs an electric propulsion system to escape an ambush by a high-tech despot. Write the adrenaline-fueled dash that defies capture.",
    "A fearless scientist builds a biomechanical interface to merge with battle gear and defeat a monstrous adversary. Write the fusion of flesh and machine that conquers fear.",
    "A visionary technologist designs a supercharged energy cannon to dismantle a villain's fortification. Write the blast that obliterates the stronghold of tyranny.",
    "A brilliant engineer creates a drone-based medic system to heal and protect against an enemy's toxic weaponry. Write the intervention that saves lives on the battlefield.",
    "A resourceful inventor develops an electromagnetic railgun to target a villain's floating fortress. Write the moment when science sends a missile of hope.",
    "A determined scientist designs a self-charging battery system to power a rebellion against an energy-hoarding tyrant. Write the uprising ignited by sparks of innovation.",
    "A fearless inventor creates a micro-drone network to infiltrate a villain's surveillance grid. Write the scene where tiny machines become agents of liberation.",
    "A visionary engineer develops a high-altitude solar collector to energize a doomed city under siege. Write the moment when light breaks through the darkness.",
    "A brilliant technologist designs an autonomous repair bot to maintain crucial defenses during a villain's assault. Write the race against time to keep hope alive.",
    "A resourceful inventor constructs a temperature-regulating system to combat an enemy who wields cryogenic weapons. Write the battle that thaws frozen hearts.",
    "A determined engineer creates a vibration-dampening shield to protect against seismic attacks from a rampaging warlord. Write the defense that quakes the enemy's ambition.",
    "A fearless scientist develops a bio-synthetic interface to enhance human abilities against a super-powered villain. Write the transformation that empowers the underdog.",
    "A visionary inventor designs a kinetic energy converter to capture and redirect enemy attacks. Write the clash that turns every blow into a beacon of hope.",
    "A brilliant engineer creates a self-organizing sensor network to detect enemy movements before they strike. Write the surveillance that anticipates danger and delivers justice.",
    "A resourceful inventor builds a neural synchronizer to coordinate a scattered resistance against a dictatorial regime. Write the strategy that unites divided forces.",
    "A determined scientist designs a regenerative armor plating to shield against relentless enemy fire. Write the breakthrough that defies the odds of battle.",
    "A fearless technologist develops a holographic navigation system to guide rebels through enemy territory. Write the map that leads to a hidden sanctuary.",
    "A visionary inventor creates a magnetic field projector to disable an enemy's heavy artillery. Write the battle where physics becomes a weapon of freedom.",
    "A brilliant engineer designs a compact fusion reactor to power an entire rebel base. Write the moment when raw energy fuels a revolution.",
    "A resourceful inventor builds an intelligent communication array to coordinate covert strikes against a powerful foe. Write the network that turns whispers into war cries.",
    "A determined scientist develops a self-learning algorithm to predict a villain's next move. Write the calculated counterattack that leaves the enemy off balance.",
    "A fearless inventor creates an eco-friendly energy harvester to empower an oppressed community. Write the revolution that springs from sustainable innovation.",
    "A visionary engineer designs a modular battle drone that adapts to enemy strategies. Write the skirmish that demonstrates evolution on the battlefield.",
    "A brilliant technologist develops a quantum computing core to crack enemy encryption and expose their plans. Write the digital breakthrough that sparks a new era of warfare.",
    "A resourceful inventor builds a solar-powered plasma rifle to challenge a villain with a reign of terror. Write the showdown that blazes with innovation and fire.",
    "A determined scientist creates a stealth propulsion system to outmaneuver a high-speed enemy chopper. Write the chase that turns silent speed into a weapon.",
    "A fearless engineer designs a self-assembling defense grid to protect a city under siege. Write the moment when architecture and technology combine to form an impregnable shield.",
    "A visionary inventor develops a gravitational lens to focus light into a devastating weapon against a ruthless oppressor. Write the clash that warps reality in favor of the oppressed.",
    "A brilliant technologist creates a force-field interface to integrate human reflexes with machine precision. Write the battle that synchronizes man and machine in perfect harmony.",
    "A resourceful inventor builds an advanced sensor drone to scout enemy fortifications in real time. Write the mission that transforms reconnaissance into salvation.",
    "A determined engineer designs a futuristic exoskeleton that amplifies strength to counter a monstrous villain's might. Write the final confrontation that redefines what it means to be human."
]

# %% [markdown]
# ## Dataset Creation
def get_writing_samples(split="train") -> Dataset:
    """Create a simple dataset from text prompts."""
    
    return Dataset.from_dict({
        'prompt': [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Write a clear, concise explanation: {prompt}"}
            ] for prompt in WRITING_PROMPTS
        ]
    })

def readability_reward_func(completions, **kwargs) -> list[float]:
    """Reward clearer, more readable writing."""
    responses = [completion[0]['content'] for completion in completions]
    
    rewards = []
    for text in responses:
        if not text:
            rewards.append(0.0)
            continue
            
        analysis = analyze_text(text, {"reading_level_target": "NORMAL"})
        stats = analysis["stats"]
        
        # Base reward
        reward = 0.0
        
        # Penalize for weak writing
        highlights = stats["highlights"]
        penalty = (
            highlights["qualifiers"] * 0.1 +   # Weak phrases
            highlights["adverbs"] * 0.1 +      # Adverbs
            highlights["passive_voices"] * 0.1 # Passive voice
        )
        
        # Small bonus for appropriate reading level
        if 6 <= stats["reading_level"] <= 10:  # Hemingway-like clarity
            reward += 0.5
        elif stats["reading_level"] > 12:      # Too complex
            reward -= 0.3
            
        # Penalize very complex writing more heavily
        if stats["readability"] == "very_hard":
            reward -= 0.4
            
        rewards.append(reward) # max(0.0, reward - penalty))
    
    return rewards

def conciseness_reward_func(completions, **kwargs) -> list[float]:
    """Reward concise writing without unnecessary words."""
    responses = [completion[0]['content'] for completion in completions]
    
    rewards = []
    for text in responses:
        if not text:
            rewards.append(0.0)
            continue

        analysis = analyze_text(text, {"reading_level_target": "NORMAL"})
        stats = analysis["stats"]

        # Base reward
        reward = 0.0

        # Reward shorter sentences (aim for 15-20 words per sentence average)
        words_per_sentence = stats["words"] / max(1, stats["sentences"])
        if 15 <= words_per_sentence <= 20:
            reward += 0.3
        elif 20 < words_per_sentence < 25:
            reward -= 0.3
        elif 25 < words_per_sentence:
            reward -= 0.6

        rewards.append(reward) # max(0.0, reward))
    
    return rewards

def active_voice_reward_func(completions, **kwargs) -> list[float]:
    """Reward active voice usage."""
    responses = [completion[0]['content'] for completion in completions]
    
    rewards = []
    for text in responses:
        if not text:
            rewards.append(0.0)
            continue
            
        analysis = analyze_text(text, {"reading_level_target": "NORMAL"})
        stats = analysis["stats"]
        
        # Base reward
        reward = 0.0
        
        # Penalize passive voice usage
        passive_count = stats["highlights"]["passive_voices"]
        if passive_count == 0:
            reward += 0.5  # Bonus for no passive voice
        else:
            reward -= passive_count * 0.1
            
        rewards.append(reward) #max(0.0, reward))
    
    return rewards


def token_length_reward_func(completions, **kwargs) -> list[float]:
    """Reward longer token counts linearly."""
    responses = [completion[0]['content'] for completion in completions]
    #tokenizer = kwargs.get('tokenizer')
    
    rewards = []
    for text in responses:
        if not text or not tokenizer2:
            rewards.append(0.0)
            continue
            
        token_length = len(tokenizer2.encode(text))
        # Linear scaling: 1 reward per 750 tokens
        reward = token_length / 500
        rewards.append(reward)
    
    return rewards

def paragraph_structure_reward_func(completions, **kwargs) -> list[float]:
    """Reward appropriate paragraph structure, especially the first paragraph."""
    responses = [completion[0]['content'] for completion in completions]
    
    rewards = []
    for text in responses:
        if not text:
            rewards.append(0.0)
            continue
            
        analysis = analyze_text(text, {"reading_level_target": "NORMAL"})
        paragraphs = analysis["paragraphs"]
        reward = 0.0
        
        # First paragraph analysis
        if paragraphs:
            first_para_analysis = analyze_text(paragraphs[0], {"reading_level_target": "NORMAL"})
            first_para_sentences = first_para_analysis["stats"]["sentences"]
            
            # Reward first paragraph length
            if first_para_sentences == 1:
                reward += 0.1  # Too short but better than nothing
            elif first_para_sentences == 2:
                reward += 0.3  # Good
            elif first_para_sentences == 3:
                reward += 0.5  # Perfect
            elif first_para_sentences == 4:
                reward += 0.3  # Good
            else:
                reward -= 0.25  # Too long
        
        # Other paragraphs analysis
        if paragraphs:
            for para in paragraphs[1:]:
                para_analysis = analyze_text(para, {"reading_level_target": "NORMAL"})
                para_sentences = para_analysis["stats"]["sentences"]
                
                if para_sentences < 3:
                    reward -= 0.1  # Too short
                elif 3 <= para_sentences <= 5:
                    reward += 0.3  # Perfect
                elif 6 <= para_sentences <= 7:
                    reward += 0.1  # Acceptable
                else:
                    reward -= 0.3  # Too long
        
        rewards.append(min(0.5, reward))
    
    return rewards

# %% [markdown]
# ## Training Configuration
output_dir = "outputs/Qwen-3B-GRPO"
run_name = "Qwen-3B-GRPO-hemingway-writer"

max_seq_length = 1024   # Can increase for longer reasoning traces
lora_rank = 64          # Larger rank = smarter, but slower

training_args = GRPOConfig(
    use_vllm = True,                # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 1,             # Decrease if out of memory
    max_prompt_length = 512,
    max_completion_length = max_seq_length,
    # num_train_epochs = 1,         # Set to 1 for a full training run
    max_steps = 1000,
    save_steps = 250, # Previously 250
    max_grad_norm = 0.1,
    report_to = "wandb",             # Can use Weights & Biases
    output_dir = "outputs",
    beta=0.1                         # Default 0.04; this increases maximum allowable KL divergance
)


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.4, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj",
        "gate_proj", 
            "up_proj", 
            "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3401239317,
)

trainer = MyS1GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs=[
        readability_reward_func,
        conciseness_reward_func,
        active_voice_reward_func,
        token_length_reward_func,
        paragraph_structure_reward_func,
    ],
    args = training_args,
    train_dataset = get_writing_samples(),
    min_tokens_thinking=200,
    max_tokens_thinking=max_seq_length,    # Large token budget
    num_ignore=4,                # If you want s1-style, ignore only 2 times
    temperature_override=4.0,     # Creative 
    min_p=0.1
)

# %% [markdown]
# ## Training
if __name__ == "__main__":
    trainer.train() 
    print("Finished training!")
    model.save_lora("grpo_saved_lora")
    print("Saved lora!")
    model.save_pretrained_gguf("model", tokenizer, quantization_method = "not_quantized")
    print("Saved gguf!!")
    import sys
    print("Success, exiting!")
    sys.exit(0)


from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 8192,
)

text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : "A rogue AI weaves a tapestry of magical simulations. Write the hacker's quest to untangle strands of fabricated myth and raw code."},
], tokenize = False, add_generation_prompt = True)

output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

print(output)
