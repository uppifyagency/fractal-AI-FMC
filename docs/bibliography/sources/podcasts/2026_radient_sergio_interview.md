# Podcast 2026 — Sergio Hernández su Fractal AI

> **Show**: Radient (radio/podcast tecnologia, conduttori Alberto e César)
> **Ospite**: Sergio Hernández Cerezo — matematico (Univ. Valencia), CEO HCSoft, fondatore Fragile Technologies, autore del paper *Fractal AI: A Fragile Theory of Intelligence* (arXiv:1803.05049, 2020)
> **Lingua originale**: spagnolo (preservata verbatim). Capitoli e tesi argomentative in italiano.
> **Lunghezza**: ~21 700 parole, ~2-2.5 ore di parlato

---

## Indice

- [0. Apertura — saluti e presentazione](#0-apertura-saluti-e-presentazione)
- [1. La tesi fondamentale: intelligenza come legge fisica](#1-la-tesi-fondamentale-intelligenza-come-legge-fisica)
- [2. La scintilla: il paper Wissner-Gross (2013) sulle Causal Entropic Forces](#2-la-scintilla-il-paper-wissner-gross-2013-sulle-causal-entropic-forces)
- [3. L'intuizione operativa: contare i futuri](#3-lintuizione-operativa-contare-i-futuri)
- [4. Il primo esperimento — il cochecito (codice in una notte)](#4-il-primo-esperimento-il-cochecito-codice-in-una-notte)
- [5. I tre pilastri dell'AGI: planning, learning, reward](#5-i-tre-pilastri-dellagi-planning-learning-reward)
- [6. Reward = personalità — la creatività dell'agente](#6-reward--personalit-la-creativit-dellagente)
- [7. Il teorema del buon controllore — bidirezionalità fisica/intelligenza](#7-il-teorema-del-buon-controllore-bidirezionalit-fisicaintelligenza)
- [8. FMC applicato agli LLM — Fractal of Thought](#8-fmc-applicato-agli-llm-fractal-of-thought)
- [9. Demo — l'evoluzione del cochecito e le prime emergenze](#9-demo-levoluzione-del-cochecito-e-le-prime-emergenze)
- [10. FMC vs MCTS — il salto di efficienza](#10-fmc-vs-mcts-il-salto-di-efficienza)
- [11. Il salto fractal — da linee aleatorie a sciami evolutivi](#11-il-salto-fractal-da-linee-aleatorie-a-sciami-evolutivi)
- [12. Il problema del labirinto — bengala vs láser](#12-il-problema-del-labirinto-bengala-vs-lser)
- [13. Il razzo con uncino — sistemi caotici e cooperazione multi-agente](#13-il-razzo-con-uncino-sistemi-caotici-e-cooperazione-multi-agente)
- [14. Atari, RAM-as-state, e l'agnosticismo dello stato](#14-atari-ram-as-state-e-lagnosticismo-dello-stato)
- [15. Cos'è 'fractal' — auto-somiglianza a tutte le scale](#15-cos-fractal-auto-somiglianza-a-tutte-le-scale)
- [16. Il terzo pilastro — la frontera caos/ordine come legge fisica](#16-il-terzo-pilastro-la-frontera-caosordine-come-legge-fisica)
- [17. Open source come strategia entropica](#17-open-source-come-strategia-entropica)
- [18. Q&A finale — infodinamica, frontera, FMC + reti neurali](#18-qa-finale-infodinamica-frontera-fmc--reti-neurali)
- [19. Il prossimo grande salto — sintesi FMC + reti neurali](#19-il-prossimo-grande-salto-sintesi-fmc--reti-neurali)
- [20. Chiusura — il borde del caos come ricetta](#20-chiusura-il-borde-del-caos-come-ricetta)

---

## 0. Apertura — saluti e presentazione

> **Tesi:** Il conduttore Alberto presenta Sergio Hernández — matematico di Valencia, applicazioni di entropia e frattali all'AI, simulazione di veicoli autonomi con comportamento intelligente emergente da semplice conteggio di futuri.

al al al al al al Bueno estamos activos qué tal vale se nos escucha a ver si está por ahí el chat y nos dicen Si el audio está bien y empezando que se escuche balanceado el audio de todos puedes darnos algunas palabras

Sergio Por favor hola hola hola hola estoy resfriado vale por el chat que que parece que se escucha todo bien César cuando quieras empezamos estupendo entonces y son las dos empezamos con la presentación Alberto Venga pues nada bienvenidos todos a radient hoy tenemos el placer de contar con Sergio Hernández que es un matemático graduado por la universidad de valencia ha aplicado conceptos de entropía y fractales desarrollo de algoritmo de Inteligencia artificial en particular ha trabajado en la simulación de vehículos autónomos que navegan a través de circuitos con obstáculos donde el vehículo decide girar basándose en el recuento de posibles posiciones futuras que es donde está la clave y demostrando así un comportamiento inteligente emergente Pues nada Sergio encantado de tenerte aquí con nosotros y si ampliar un poquito más información sobre esta Pues nada estamos aquí para escucharte Pues así como resume nos diría un poco que la idea del algoritmo es que básicamente que la

---

## 1. La tesi fondamentale: intelligenza come legge fisica

> **Tesi:** Le reti neurali stanno simulando il **principio di minima azione** (Lagrangiano). Sergio si chiede: e se applicassimo l'**altro principio fondante della fisica**, quello di massima entropia, per costruire qualcosa di parallelo? Stessi formalismi, mondi paralleli.

que la física y los procesos que pasan en la inteligencia Aunque parezca que son dos cosas que están totalmente desconectados realmente son son son lo mismo las mismas fórmulas que funcionan en un sitio funcionan en otro y son totalmente digamos paralelos mundos paralelos Y entonces en este caso ándose en un principio del segundo la segunda ley de la termodinámica que la entropía siempre crece se puede llegar a crear un algoritmo que realmente te genera un comportamiento inteligente que que sería esperable simplemente por el hecho de que está simulando una ley física fundamental y eso es un poco lo lo lo increíble del tema

no similar a Como por ejemplo las redes neuronales cuando aprenden realmente están simulando un sistema físico y lo están resolviendo por por el lagrangiano y están utilizando realmente en el fondo fondo fondo están utilizando el principio de mínima acción y como está basado en un principio de mínima acción y está bien aplicado el algoritmo funciona y las redes aprenden no

Pues un poco el el Paralelo con el segundo con el segundo principio fundamental de la física sería eso qué pasaría si tiramos del otro principio de la máxima entropía y construimos algo parecido a lo que tenemos como una red neuronal Qué es lo que obtendríamos no que qué inteligencia obtendríamos es equiparable es complementaria a la que obtenemos con redes neuronales un poco todo eso el campo en el que yo me muevo entonces Estableciendo como un primer Punto de partida eh podemos decir que al igual que las redes neuronales están simulando una eh ley o teorema físico eh tu planteamiento sobre esta inteligencia emergente basada en entropía eh básicamente que partimos de también de un principio físico y a través de esa simulación o esa eh [Música] esos cálculos que que hacemos sobre eh Cómo vemos la entropía futuro podemos ver que hay como cierta inteligencia que está emergiendo de estas simulaciones sí básicamente toda la idea se basa en un paper que salió en el 2013 que

no sé si estáis viendo en la pantalla ahora o no pero que y lo podemos poner César Okay vale

---

## 2. La scintilla: il paper Wissner-Gross (2013) sulle Causal Entropic Forces

> **Tesi:** Il momento di illuminazione di Sergio: tutti i sistemi naturali seguono la 2ª legge della termodinamica massimizzando l'entropia. Perché l'intelligenza non dovrebbe seguire la stessa legge, ma applicata a un orizzonte temporale **lungo** (5 secondi invece che il tempo di Planck)? Il paper formalizza questa idea ma le formule sono intrattabili nella pratica.

este fue el momento en el que yo digamos vi la luz No yo conocía las redes neuronales y todo eso estaba muy bien pero me faltaba un poco esa base física para decir ahora lo entiendo ahora tengo una teoría tengo una manera de pensar en Qué es la inteligencia Cuántas patas tiene Cómo se hace cada una cómo se y y eso a mí me lo dio este paper que decía que básicamente decía

Bueno a ver si todos los sistemas naturales siguen la segunda ley de la termodinámica evolucionan en el tiempo de manera que la entropía se se va maximizando Por qué el comportamiento intelig inteligente no va a seguir la misma ley Entonces se plantea un poco se replantea la segunda ley la termodinámica diciendo bueno Y si yo defino la entropía no como se define en física el macroestado y tal si

no Y si yo me imagino que tengo un estado y que dejo pasar un segundo y me imagino todos los posibles estados en los que mi estado inicial se puede propagar digamos no todos los posibles caminos que puede seguir en un en un segundo si yo ahora calculo la entropía de eso de qu esa mancha de dónde puede estar en el futuro esa predicción tiene una probabilidades puedes calcular entropía y yo puedo saber si el los futuros que le esperan a esa partícula son más diversos o menos diversos midiendo la entropía de de de de esa distribución Entonces si tú lo te lo planteas así como que es entropía a futuro entonces la física Solo dice que a un tiempo muy pequeñito de plank los sistemas evolucionan de manera que su entropía a futuro en ese saltito es máxima y eso te da la segunda ley de la termodinámica

Y entonces este hombre se planteó lo siguiente y si yo crease un sistema que fuese capaz de predecir a más largo plazo y que entonces optimizas la entropía O sea la cantidad de futuros diferentes que va a tener a un plazo mucho más largo si un sistema pudiera ser inteligente y pudiera predecir eso a 5 segundos vista tú verías que siguiendo la misma ley de la de la termodinámica pero aplicado

no a tiempo de plan sino a 5 segundos la diferencia sería que en lugar de comportarse como una piedra que se cae o un gas que se expande se comportaría como un [ __ ] inteligente que va por ahí sorteando obstáculos eso es un poco la hipótesis de de de de entrada del del paper que suena como alucinante no O sea lo piensas Y di tú o sea es posible que con esa cosa tan sencilla tan sencilla bueno las fórmulas son de infartos

no pero la idea es muy sencilla un vistazo a la fórmulas a ver qué tal la fórmula ya os digo Bueno a nivel teórico está todo bastante bien todo lo que tú quieras pero la entropía a futuro sería la integral esta que veis aquí de una probabilidad logaritmo de una probabilidad tara pero esa probabilidad su vez de la integral en todos los posibles caminos empieza a tirar del hilo de Qué significa cada integral cada integral y realmente

no hay manera de de tratar esa fórmula no O sea que la idea muy bonita pero al final lo que te encuentras es que todo lo que involucra calcular entropía en física y en todo es tremendamente complicado la entropía es una cosa casi imposible de de tratar no Y entonces el el autor lo hace lo mejor que puede con sistemas muy sencillo hace simulaciones con Monte Carlo y trazando muchos caminos haciendo la media de la probabilidad de que pase Y así más o menos para calculando Y de alguna manera consigue unos vídeos que tú dices parece que está haciendo algo

no pero es muy difícil de verlo porque hacer estas integrales de aquí realmente no se puede entonces se aproxima de una manera muy muy muy muy rudimentaria

Vale entonces aquí el el Esto fue como la primera piedra Por así decirlo en la cual tú te inspiraste pero claro te enfrentas a este problema que el cálculo de la entropía es bastante complejo Y qué es lo que se te enciende qué es lo que se te ocurre para poder ir e pues el tema Fue bastante curioso yo cuando me enteré de este paper fue por una publicación en un en un blog que yo seguía de ciencia y ahí te lo explicaban un poco resumido no como más poético no más para el público entonces básicamente no te decía toda esta fórmula sino que te decía eh la inteligencia

---

## 3. L'intuizione operativa: contare i futuri

> **Tesi:** La rivelazione semplificatrice arriva da un blog divulgativo: *l'intelligenza è scegliere l'azione che apre più futuri possibili*. Esempio concreto: studiare vs. lavorare apre quantità diverse di lavori futuri. Tradurre questa idea in algoritmo è semplice: simulare 100 cammini casuali per ogni decisione, contare gli stati raggiunti, scegliere l'azione che produce maggior dispersione (entropia).

se basa en elegir en cada momento aquella acción que te Abre más posi futuro como diciendo si yo puedo elegir Pues siempre pongo el mismo ejemplo yo puedo elegir estudiar o puedo elegir irme a trabajar conforme termino el colegio Las dos cosas están bien pero si tú miras a futuro Pues en un caso vas a tener menos oportunidades menos posibles empleos que puede al final tener vamos a medirlo en empleos No si ha hecho una carrera pues puede tener muchos empleos desde más sencillo a más complejo y si

no ha hecho la carrera tien menos empleo si lo mira solo con esa métrica digamos dirías pues tiene más entropía estudiar la carrera porque me abre más futuro entonces claro Aunque todo eso se traduce en esa ecuación tan fea al final es contar futuros diferentes y eso no es tan complicado yo Puedo imaginarme un sistema y decir voy a simular uno de estos posibles caminos que salen Aquí eso simplemente es decir tomo una acción de todas las acciones que puede tomar el sistema apretar un botón mover una palanca a un a un Word model le pregunto Oye qué pasará después si pulsar esa palanca y esta situación el Word model el que sea un simulador o lo que sea un predictor del siguiente estado me diría dónde voy a estar vuelvo a tomar otra otra otra acción aleatoria y yo puedo construir lo que aquí en este en este dibujo veis como todos esos caminitos que suben para arriba en el cono Entonces yo estoy en el punto de abajo de ese cono estoy calculando uno de esos caminitos que sube para arriba y yo puedo llegar hasta arriba y decir mira este camino termina en este punto en la tapadera de arriba

no Entonces yo como como yo me lo imaginé fue voy a calcular 100 caminitos de esos aleatorios y voy a ver dónde acaban Y si acaban muy cerca hay poca entropía el cono es pequeño y si acaban muy lejos la entropía es muy grande y como me lo imaginé así de repente todo fue muy sencillo Porque si tú te estás planteando tú por ejemplo vas conduciendo un coche tiene un grado de libertad que si presiono el volante hacia la izquierda hacia la derecha básicamente yo lo puedo reducir a dos AC que tengo o presiono hacia la izquierda o presiono hacia la derecha eso me abre un arbolito con Solo dos ramas qué ocurre si presiono la izquierda o a la derecha yo puedo predecir mi siguiente estado con mi World model y ahora en esos dos puntos es donde yo hago este este truco me imagino 100 caminos y calculo la distancia digamos media de los puntos finales me imagino otros 100 caminos en el otro lado y calculo la distancia media el que tenga la mayor distancia media el mejor el que tiene más entropía me abren más futuros diferentes Entonces eso lo lo hice en una noche porque

no pudiste dormir esa noche no

---

## 4. Il primo esperimento — il cochecito (codice in una notte)

> **Tesi:** Sergio implementa l'idea in **una notte** (prima ancora di leggere il paper). Un'auto in 2D con 2 gradi di libertà (sterzo + accelerazione). Aspetto chiave: **il motore si è acceso da solo** — l'algoritmo, senza alcuna istruzione, scopre che muoversi apre più futuri di restare fermo. Senza apprendimento. Senza training. Solo simulazione + entropia.

sí dormí muy bien pero fue porque no me había leído el paper si me llego a haber leído el paper antes de ponerme a hacer el código me habría replanteado todo esto y habría dicho madre mía Cómo simplifico esta ecuación y me habría metido en un callejón sin salida Pero como me tiré a ciega como se me yo me lo imaginé así

Y eso muy sencillo de hacer si tiene un simulador de cualquier cosa simular 100 caminos y se programa en un momentito No si tiene un simulador físico a mano en mi caso no tenía y lo program también esa noche aquí hay algo muy interesante que has comentado al principio y que me parece reseñable recalcarlo Y es que al final estamos viendo si tú te preparas ese simil que decías Oye siud te abres más camino no solamente si estudias no esto se puede dar en cualquier ámbito

es decir a mayor preparación mayores posibilidades con lo cual estás aumentando tu entropía y y aquellas selecciones que tomes que maximicen tus opciones al final son las más inteligentes a tomar no sí lo curioso es que el principio cuando tú lo aplicas así yo lo que hice fue un cochecito que podía avanzar y precisamente era el volante que podía darle hacia la izquierda hacia la derecha y tenía un acelerador que era también un botón para

no liarme no O sea tenía dos grados de libertad acelero más acelero menos giro izquierda giro derecha y en principio El cochecito aparecía parado en medio de la pantalla digamos no Y claro no hay ninguna instrucción ahí o sea el algoritmo es ciego y y lo primero que me planteé qué hará

Y claro conforme le di al algoritmo el coche arrancó y se puso andar y a recorrer el circuito y yo dije de dónde ha salido eso o sea por qué se le ha ocurrido arrancar el coche claro Endo ciertas capacidades no de un principio tan simple claro estaba emergiendo un comportamiento inteligente qué es lo que estaba ocurriendo el coche pensaba si no puso el acelerador en el siguiente segundo voy a estar aquí mis posibles estados son uno es lo peor que me puede pasar quedarme quieto a nivel de entropía porque puedo predecir mi futuro perfectamente

no he pulsado el acelerador voy a estar en el mismo sitio Si tú puedes predecir tu futuro la entropía es cero porque la entropía es la incertidumbre de tu predicción vista de otra manera ahora si yo pulso el acelerador a continuación en el siguiente segundo se abre un cono donde yo ahora puedo acabar en un montón de puntos se ha abierto el cono tengo más área tengo más separación entre los puntos tengo más entropía entonces siguiendo ese proceso Tan sencillo el [ __ ] decidía pulsar el acelerador Y eso salía solo y cuando

Lu iba avanzando y se encontraba que había una curva si no giraba hacia la izquierda sus futuros se reducían porque se chocaban Y se quedaban pegados digamos en La Curva y solamente porque se reducía la distancia entre los posibles futuros y hacia la izquierda No pues decidía irse hacia la izquierda y entonces solo con ese principio tan tontito y esa esa aproximación programada en una noche el coche daba vuelas En un circuito y tú le podías poner obstáculos le podías poner trampas daba igual era total podías toquetear se enfrentaba a un circuito que ni siquiera conocía

Aquí no hay ningú aquí no hay ningú no hay nada de aprendizaje No hay nada de aprendizaje en lo que es este algoritmo o sea que claro no lo había visto nunca qué ocurre que realmente aquí lo primero que te asalta es bueno este chico está haciendo trampa porque dice que no hay aprendizaje pero tiene un un World model tiene un simulador que le dice dónde va a estar el coche a continuación

Claro en el mundo real tú tienes que tener un sistema de learning que te aprenda a decirte dónde va a estar el coche en el siguiente Segundo si pulsa estos botones O sea que en la vida real tú necesitas un Word model que lo habrá hecho con una red neuronal que lo utilizas para simulador para el que te va a hacer el planning de que tienes que hacer

---

## 5. I tre pilastri dell'AGI: planning, learning, reward

> **Tesi:** Aprendere a predire (world model — l'unica scorciatoia che Sergio si è preso, scrivendolo a mano), pianificare lungo termine (FMC stesso), e **valutare** (reward function). Il reward è il punto critico: definisce la personalità dell'agente. Senza reward esterna l'agente diventa un *common sense bot* che si muove ma non vuole nulla — sopravvive ma non ha utilità.

claro aquí ya empezamos a hablar un poco de las diferentes patas que puede tener la Inteligencia artificial hablamos el tema del planning hablamos del tema del learning y hablamos también de del redward no que faltaría A lo mejor ahí ver un poco e Cómo patas eh la estás abarcando con tu aproximación sí

lo que me encontré fue lo siguiente si yo tenía un buen simulador de mi sistema que en este caso era perfecto porque lo había hecho yo no no no no no era prendido ahí hacía trampa pero claro yo quería comprobar Cómo funcionaba el algoritmo mío no el uno de learning por ahí no entonces eso siempre se hace así se le da un simulado perfecto para ver cómo funciona el algoritmo de Planet pero necesitaba siempre añadirle un tercer ingrediente Y es que yo le tenía que dar la ruar en el caso de que yo quisiera un un car si si si tú al Car

no le daban ninguna ruar ninguna recompensa no no le definía como de bueno en un estado como de malo lo que hacía era lo que te he dicho encendía el motor y empezaba a andar Simplemente porque quedarse quieto tiene menos entropía O sea se movía por por medio de la de la pista pero no quiere hacer nada simplemente no quiere quedarse quieto quiere tener acceso a muchos sitios

no le gusta meterse en un callejón sin salida digamos es como adquiere sentido común pero no tiene ninguna preferencia por ir a un sitio O sea no le puede educar no le puede hacer que hagan nada no tienen ninguna utilidad Entonces nosotros no funcionamos así nosotros no decimos me voy a esta habitación porque es más grande y tengo más posibles futuros di me quedo honesta porque están poniendo una película en la tele o lo que sea y tú tienes una atención que me interesa

no Claro si le quita eso le has quitado totalmente la gracia a cualquier inteligencia no O sea tiene una cosa mecánica que ha aprendido a predecir el futuro que sabe Ahora predecir tuus acciones a largo plazo usando un algoritmo de este estilo Pero ahora tiene que decidir qué quiere hacer y eso en una en una Ai lo tendría que hacer el propio algoritmo O sea si tú quieres crear una egi

tienes que resolver los tres problemas aprender a predecir aprender a planear a largo plazo y aprender a valorar dado dos posibles futuros Cuál es mejor que el otro de alguna manera entonces A eso se le estaba dando a mano Entonces igual que todo lo que aprender a nivel de física está conectado con el principio de mínima acción el Cómo planear está conectado con el principio de máxima entropía pues lo que llevo muchos años dándole vuelta es cóm Imagínate que tú tienes un robot que le has puesto un algoritmo dentro pero

no tiene datos está totalmente vacío y tú lo enciendes y y se encuentra en una isla desierta por ejemplo y no sabe ni que es un robot O sea tené que empezar a como un bebé no tú podrías hacer que ese robot al final prosperar y no sé qué si no decide qué quiere hacer No claro se va a quedar ahí quieto y ya está se va a quedar tomando el sol digamos no pero claro si dice

no no Yo quiero mantener mi energía alta voy a construir algo que me recargue la energía porque si no me voy a pagar dentro de una semana para eso él tiene que planear pero para planear tiene que tener un objetivo Cómo define tú los objetivos de un una inteligencia genérica Y ahí surgen objetivos de manera natural en la interacción entre los dos sistemas porque por ejemplo ese robot querrá aprender sobre su entorno querrá saber poder predecirlo tendrá que primero explorarlo y entonces le tiene que dar al algoritmo de planning de alguna manera Oye me gusta explorar Entonces qué haría de una manera genérica pues podría decir a ver Yo sé predecir mi futuro si esa red neuronal también tuviese un predictor de su propio error que simplemente la entrena a predecir la los de tu predicción tú ahora podrías mirar tu isla y decir si voy debajo de la palmera sé lo que va a pasar pero si voy allí que hay una cosa que nunca he investigado hay una piedra No sé lo que hay debajo lo que sea o hay una planta extraña yo

no sé lo que va a pasar ahí yo puedo digamos catalogar de todas las cosas que tengo enfrente las que conozco las que no conozco y las que medio conozco Y de alguna manera el algoritmo de learning puede decirle al otro Oye esto que medio lo conozco te gusta si va un sitio que sabes predecirlo al 50% Haz que te guste porque de esa manera empezará a andar y empezará a visitar sitios donde yo

no sé predecir bien lo que va a pasar y en muy poco tiempo Aprenderé a predecir todo lo que puede pasar en la isla entonces parecen sinergias de ese estilo y está muy bien pero por ejemplo hay otras cosas como por ejemplo ese robot debería de tener un objetivo que es mantener su energía alta pero eso él no lo puede aprender porque en el momento en que se le acabe la pila la primera vez que se acabe la pila ya está muerto él

no lo va a poder aprender entonces en nuestro caso eso ya viene programado genéticamente Y tenemos cosas que nos gustan de nacimiento como el chocolate digamos no porque azúcar te las tenemos en bebidas dentro de nuestro genoma para poder eh satisfacerlas No sí tenemos un sistema que no que comer Tenemos un sistema muy complejo dentro de nuestro cuerpo que nos dice lo que nos gusta y lo que

no nos gusta y está bastante bien afinado si haces lo que te gusta Vas a vivir tiempo digamos entre comillas no Es como te dan como un heurístico de qué tienes que hacer para sobrevivir porque los que no han hecho caso se han muerto hace tiempo y no son tus padres no digamos O sea que entonces tú tienes que en ese tercer Pilar te vas a encontrar con algunos objetivos que son intrínsecos a la colaboración entre los dos primeros Pilares que está muy bien Te alegran la vida hacen que recién encendido y haga cosas por ejemplo como hemos visto aquí el el intentar hacer que la diversidad sea máxima de Su futuro lo puede entender como un goal que está ahí embebido dentro de del propio algoritmo Y

esa serie de dos o tres goal que ya te dan de fábrica te vale bastante necesita tener algún goal del estilo genético que me ha venido por por un proceso previo de selección natural o de algo y luego evidentemente va a tener que tener otros que generes tú durante tu vida y que digas me gusta más la fresa que el plátano o lo que sea no eso lo has generado tú durante tu vida y hace que tú en tu día a día vaya a comprar fresa O sea tú haces lo que haces Y estudias lo que estudias y tal porque antes digamos ha florecido ese gusto en ti si no te gustase no lo haría Y esa

---

## 6. Reward = personalità — la creatività dell'agente

> **Tesi:** Tutto in FMC è meccanico tranne la funzione di reward. **Definire la reward = definire la personalità**. È l'unica parte creativa dell'algoritmo. Senza reward, il robot resta vivo ma non vuole nulla; con reward composta multi-obiettivo (energia × salute × goal) emerge un comportamento biologicamente realistico — l'agente si muove come un essere vivo, equilibra istinti, sopravvive in maniera saggia.

esa función de recompensa que tú creas es lo único creativo que hay en el algoritmo todo lo demás es mecánico entonces muy gracioso porque definiendo la recompensa defines totalmente la personalidad de de ese agente es donde está la la la clave de todo no y donde más tiempo pasé O sea que al final de estas tres patas el redw Parece ser el punto clave sí es como si tuviera aprendizaje redes neuronales planear a largo plazo v y la tercera que sería digamos técnicamente es la función de riguar pero es la que te da la personalidad Y algo así como como la conciencia o el alma o tu forma de ser

no sé Ahí ya entra dentro de un poco de lo esotérico pero todo eso encaja Digamos si yo pudiera modificar mi función de riguar yo sería otra persona totalmente

es lo que me Define no entonces la construyes durante tu vida Cómo se construye es o sea cómo se hace No eso es lo que ando dándole vuelta para resolver digamos la tercera patita no

vale Y recapitulando un poco hemos hablado de artificial hemos tocado un poco lo que sería el teorema del buen controlador estamos hablando de ese War model e Cómo fue un poco indagar en esta parte del buen controlador En qué consiste Y por qué puede ser tan importante bueno como como digo yo primero empecé por el tema de la entropía me llama mucho la atención no y luego viendo que era un principio básico de la física y que la las redes neuronales se basaban en el otro Claro ya te entra la duda de decir bueno esto tiene alguna base o simplemente que se le parece o o que como no nada más somos martillos solo vemos Púas no y Y entonces eh recordé que hay

---

## 7. Il teorema del buon controllore — bidirezionalità fisica/intelligenza

> **Tesi:** Teorema della teoria del controllo: **qualunque controllore di un sistema complesso deve contenere una simulazione di quel sistema**. Quindi il world model del cervello *non è una coincidenza* — deve essere isomorfo alle leggi fisiche del mondo reale. Conseguenza: pietre che cadono = intelligenza a tempo di Planck. Le leggi della fisica sono il limite a τ→0 dell'intelligenza.

hay un teorema en en teoría de control que es muy curioso porque dice que cualquier controlador de un sistema complejo ha de contener una simulación del sistema que intenta controlar O sea que tú no puedes hacer un controlador que no sea digamos de alguna manera sea copia sea una simulación de lo que está controlando y claro cuando tú piensas en la inteligencia la definas como la definas al final el uso que que tú le das es manipular tu entorno para tu beneficio que puede ser lo que sea

no pero tú estás manipulando el entorno eres un controlador de tu entorno que sigue las leyes físicas tú quieres variar el curso del río tú quieres que no te dé el viento tú quieres que se caliente la habitación Y entonces qué más tú tienes un simulador de cómo funciona tu mundo que lo que nosotros siempre llamamos un Word model y según ese teorema ese Word model tiene que contener una simulación del mundo real o sea que realmente es un Word model

no es una cosa que se ha inventado el cerebro y que da la casualidad que funciona igual que el universo no no tiene que ser una simulación de las leyes físicas del universo porque es la manera óptima más sencilla de hacer un controlador de ese sistema Entonces

es totalmente esperable que cuando tú indagas En cómo qué algoritmos te funcionan en temas de Inteligencia artificial te des cuenta que están totalmente relacionados con uno de Física Pero en este caso sido al revés siempre un físico que conocía mucho un problema se ha imaginado que pasaría si lo podría adaptar a inteligencia Y así surgió en la neuronales y son los premios Nobel de física de este año no y este paper

y este paper también es un físico pensando en tema de física y diciendo Oye no se podría extrapolar esto la inteligencia entonces estendo que desde la física estamos viendo estos comportamientos inteligentes pero se puede dar también eh Esa bidireccionalidad sí de hecho el el el tema que se se se

se difumina totalmente la frontera porque tú dices una persona Es inteligente porque está planeando a futuro está viendo la entropía de sus futuros está haciendo Está siguiendo la la segunda ley de la termodinámica donde esos 1000 futuros que se está imaginando evolucionan en el tiempo en su cabeza siguiendo haciendo lo mismo que haría un gas pero es que un gas hace eso a tiempo de plank entonces la física

no es más que inteligencia a tiempo de plank previendo las cosas a un tiempo de plank Qué diferencia hay o sea las piedras caen para abajo porque son muy inteligentes pero solo ven el siguiente microsegundo no saben que se van a dar un golpe cuando cuando tú haces tender el tiempo al que tú planeas lo haces tender hacia cero lo que tienes son las leyes de la física Esa es la gracia

no Según a Qué escala lo mires lo considera inteligencia porque está and a 5 minutos vista eso es claramente inteligente pero si simplemente Cae por la ladera la piedra dice buah Este no es inteligente yo habría ido Por otro sitio porque yo estoy viendo a más largo plazo y veo que por aquí llego antes la piedra solo ve el siguiente microsegundo y va a cuesta abajo que es lo que lo mejor que puede hacer No gradios desten

no va hacia abajo que digamos que es la solución más evidente no si no puedes planear a largo plazo y entonces se difumina totalmente y luego cuando tú ya te planteas cosas de tiro de una un chat gpt Cómo funciona el Transformer todo ese tipo de cosas y tal yo no puedo evitar verlo de la misma manera yo me imagino que

---

## 8. FMC applicato agli LLM — Fractal of Thought

> **Tesi:** Trasferire FMC alla generazione di linguaggio: ogni frase è un cammino, ogni token un tick. Servono due ingredienti non banali: una **distance metric** tra frasi (embedding) e una **reward** che valuta la qualità di una frase incompleta. I modelli razionatori (o3, ecc.) percorrono **un solo cammino** — Sergio prevede che si scontreranno con un muro: cammini lineari non trovano mai i sentieri stretti dei labirinti complessi.

tú estás generando una frase a base de generar token que es dando pasitos y tú cada pasito lo estás dando de manera que el su probabilidad sea la máxima está haciendo lo mismo que la física está dando pasitos y generando el camino que va cuesta abajo en cada cada momento es física si tú Bueno física me refiero es lo mismo que hacemos en física si tú pudieras ahora eh generar ritas de token líneas de token que se bifurca y en un momento dado puede elegir dos palabras tres palabras y pudiera elegir un fractal de frases como como lo que vas a decir y le pudiera aplicar el mismo algoritmo que estamos hablando aquí imaginándonos que cada frase es una de estas uno de estos caminos y cada pasito Añadir un token y que hay una distancia entre frases y que hay una puntuación de frases según tengan más sentido o menos tú puedes aplicar el mismo algoritmo y generar un cono de frases que se autoajusta de manera que hay más frases que pasan por las zonas que más sentido tienen lo mismo que aquí habría más caminos que pasan por las zonas donde hay más ruar porque le atraen No pues tú podrías imaginármelo cabeza es lo mismo pero ha cambiado el lenguaje y de repente está hablando de física de planning o de planning dentro de un algoritmo que produce token que realmente los ha aprendido pero fíjate que en ese caso lo que ha aprendido es solo a dar el siguiente paso apr decir el siguiente token es lo mismo que decíamos en física tú aprendes al siguiente pasito en los coches tú aprendes a dar el siguiente pasito y ahora planeando veo a más tiempo vista No pues en una red normal pasa igual Entonces ahora ahora mismo estamos en eso los modelos digamos con pensamiento y tal

no llegan a ser eso pero internamente ya empiezan a tener esa diferentes frases que se van creando que se bifurcan que se anulan que que hacen un juego una especie de de de de algoritmo evolutivo y y unas desaparecen se copian a la otra se bifurcan que es exactamente lo que digamos hace este algoritmo por eso digamos a la misma vez es muy particular se basa en una ley física y es para planning pero a la misma vez se puede adaptar a todo igual que una red neuronal es para aprender pero la puede utilizar para razonar son todo como son conceptos muy elásticos con solo una ley física podría abarcar toda la física Aunque hay una parte de la física que se ve mejor con la otra ley física aquí pasa igual con una red neuronal puedo llegar a hacer planning a cierto nivel razonar pero con un algoritmo de razonar puedo hacer mucho más y puedo incluso unirlo con el otro y hacer un solo algoritmo que sería un poco en física sería utilizar los dos principios a la vez para resolver un problema que

no se hace nunca básicamente

vale o sea que aquí empezamos a hablar un poquito eh de la parte más convencional ahora mismo de Inteligencia artificial que son los glm eh Cómo funcionan estos nuevos algoritmos de Transformer donde van eh prediciendo el siguiente toque

y por lo que nos comentas que se podría eh aplicar este algoritmo basado en entropía viendo un poco Cómo se ramifican en vez de ya camino serían diferentes palabras no y qué tendríamos que tener aquí en cuenta e Cómo se maximiza la entropía o cómo minimizamos esa los minimizar la los va ap parte eso quiere minimizar la los es con respecto al siguiente token eso tienes que minimizarlo Y tú Tú ya parte de una red neuronal que tiene su pretraining que que funciona ese no es tu problema

el problema básicamente si si lo dice en lenguaje natural sería H intenta predecir como siguiente palabra aquella que te abre un amplio abanico de siguientes párrafos diferentes y que tiene sentido o sea yo empiezo una novela y quiero que la primera frase sea muy sugerente en el sentido de que madre mía puede pasar cualquier cosa a continuación no entonces esa intriga de puede pasar cualquier cosa todo está abierto podría en la siguiente página pue pasar cualquier cosa Eso es lo que le da digamos entropía al trozo que tú has leído

no puedes predecir Su futuro se abre mucho futuro igual que que en el otro igual que cuando hablas de un car tú quieres mucho futuro y por supuesto quieren que sean futuros buenos o sea que también tiene una rigual claro

Entonces el el mayor problema que tienen este tipo de cosas es cómo asigno una ruar Cómo asigno una distancia a dos frases que estoy yo completando a dos respuestas de una llm entonces tú puedes pensar bueno distancia no puedo usar los caracteres I no puedo usar los códigos de los tokens ni nada de eso eso no tiene sentido tendría más sentido usar una especie de de de Word to Back de de poner las palabras en sitio y ver la distancia eso sería lo ideal

no pero bueno si tú puedes [ __ ] Trozos de frases frases enteras lo que sea y hacerle un embedding a toda la frase ese embedding viene siendo un Word to Back básicamente te da un vector con el significado de esa frase Y entonces en ese espacio O sea si tú le haces el embedding a los textos que llevas generado puedes calcular la distancia entre los embedding y tienes una distancia totalmente usable Si dos cosas dos conceptos son muy diferentes en dos respuestas diferentes su en Bed estarán muy lejos y si es lo mismo dicho de manera similar estará muy cerca igual que pasa en los edin de las palabras los edin de las palabras pues se se utilizan técnicas para que Rey menos hombre más mujer igual a Reina está haciendo que funcione como un espacio vectorial de significado cuando tú ahora miras la distancia entre rey y reina es pequeña pero si mira entre rey y caballo es más grande

no O sea si tú haces lo mismo con frases ya tienes tu distancia y luego te falta la arriu O sea que tú tienes que ver que una frase una respuesta que te está dando una llm si es mejor o si es peor pero aún no está acabada entonces claro el decir hay más posibles finales está bien tienes más salidas posibles si te has metido en una atolladero lo digamos

no pero tú quieres saber si lo que ya ha escrito es mejor que lo que ha escrito otro de otra de las versiones no yí entra el segundo problema de cómo evaluar la calidad de una frase a medio general y Claro tú ahí puedes entrenar una red para que muchos ejemplos Pero al final cuando entrenas de esa manera realmente lo que estáis aprendiendo como un loro a que este tipo de respuestas suelen acabar bien pero

no está realmente resolviendo el problema no entonces cómo puedes tú determinar si una frase tiene mejor pinta que otra entonces ahí hay diferentes técnicas básicamente puedes usar en beding y cosas así te metes en líos O puedes decir Bueno mira cada uno de los token en su momento Tuvo una probabilidad algunos que elegiran de 01 y otros que elegiran de 03 Yo podría hacer una mezcla de todas esas probabilidades de las palabras que llevo generadas hasta ahora de los tokens y eso me podría dar una especie de de aproximación de cómo de buena está haciendo la cosa pero de nuevo

no es del todo lo correcto no O sea estás viendo que la frase hasta aquí es más predecible más fácil de predecir entran más en lo predecible que la otra Pero igual hay una genialidad y por eso no es predecible no entonces ahí entra en el problema duro de cómo y lo evalúo no entonces el el

el algoritmo es muy bonito pero cuando lo lleva a la práctica en según qué caso puede no estar no estar claro lo lo que es cada cada punto No eso un poco donde donde entrar la dificultad de adaptar El algoritmo a un caso o a otro

Pero cuando se adapta va muy bien eh eso está claro entonces para para redes neuronales hemos hecho intentos pero todavía estamos en ello para que digamos de repente diga gu ya sabe planear claro habrá que dar ciertos pasos y veremos Dónde dónde estaremos en un futuro

pero me gustaría hablar un poco más que del futuro del pasado hemos empezado un poco a conocer tu trabajo ver este nuevo enfoque basado en entropía pero lo que nos gustaría ya nos ha hablado de los cochecitos nos has hablado de estas decisiones

podemos ver cómo funciona eso realmente sí seguir viendo la pantalla verdad A ver seguí viendo la pantalla todo venga lo ponemos ahí yén un poco sobre esto

Bueno est es un vídeo que hice eh básicamente para para dar charla y poner trocitos no y aquí fui poniendo trocitos de vídeo que fui generando básicamente para ver el camino que fui siguiendo porque al final cuando ve el algoritmo terminado parece que no no lo entiende Si no ve el Caminito no Y entonces mira os voy a poner el principio que sería Esto fue lo que program la primera noche digamos Este vídeo del día siguiente y es lo que os digo eh lo voy a parar y primero explico mira aquí esto sería el primer coche que fijaos que era un triangulito simplemente el acelerador siempre estaba a tope y él solo podía decidir si eh su grado de libertad era mover el volante un poco más a la izquierda un poco más a la derecha o sea que era digamos o uno o menos uno digamos solo tenía dos acciones una vez que tomaba una de ellas yo dibujaba 100 futuros posibles que por ejemplo los rojos son si primero elijo a la derecha y los azules y primero elijo a la izquierda eso me daba esta nubecita de puntos que habían rojos y Azules y calculaba digamos la distancia media entre rojos la distancia media entre azules básicamente era eso Y proporcional a esa distancia media los que más distancia media tenían hacia ese sitio tendía a girar No si había más distancia media en los rojos giraría un poco hacia el lado rojo O sea que el algoritmo el más sencillo que he comentado o sea esto fue el germen de todo

no fue el inicio donde explota sí Esto fue Esto es lo que ocurrió hasta antes de leer el p digamos no hice primero esta prueba rápida y aquí no estás programando el comportamiento del coche para nada para nada para nada O sea que para nada aquí ya aquí el acelerador si estaba estaba pulsado con lo cual yo esperaba simplemente que fuese yendo hacia la izquierda o la derecha cuando le puse el acelerador fue cuando pasó lo que os conté de que empezó aceleró el sol

no luego ya me acostumbré y lo veía normal y bueno la siguiente prueba que hice fue ponerle rozamiento y cosas de esas así para ver si era está tan listo tan tonto porque Claro si no ve una cosa difícil no un algoritmo tan raro tan Alien si no lo ves funcionando Y ves lo que hace no no no no sabes evaluar tú mismo teniendo este algoritmo que lo que está haciendo es calcular futuros probables eh pueda llegar a resolver este tipo de problemas

no bueno Y esto a mí me sorprendió eh el primer día me sorprendió luego me acostumbré pero luego me siguió sorprendiendo con los siguientes problemas O sea que no paro aquí la cosa o sea el algoritmo era muy muy muy muy muy potente o sea hacía unas cosas que que a fecha de hoy todavía las veo y digo madre mía cómo pudo hacer eso el algoritmo si le puse 500 futuros no O sea si no llevaba nada no y realmente es muy potente luego

el siguiente paso que di fue a ver lo anterior es lo que yo te estaba comentando de que tú solamente buscas muchos futuros posibles Cuanto más mejor que sean muy distantes pero no me preocupo de si mi energía y mi energía baja no me preocupo De nada de eso no

Y claro entonces pues simplemente se va moviendo Y entonces claro lo siguiente que hice fue bueno en lugar de contar los puntos cada punto cuenta uno pues voy a contar cada punto puntuando solo con la longitud que ha recorrido el coche los puntos más lejanos suman más que los más cercanos Y al hacerlo de esa manera lo que está haciendo es que elija

no la opción que tiene solo más futuro sino más futuros más largos le está añadiendo ya una personalidad ya le está diciendo Oye distancias largas mejor que distancias cortas Entonces le gusta correr más no Claro ese fue el germen de que me fui dando cuenta de que realmente modificando ese parámetro esa función podía hacer lo que quisiera y entonces en los siguientes digamos meses Me dediqué a imaginarme caso imposible de resolver y darle una reard de si lo consigues te doy un caramelito y lo resolvía claro lo que estás haciendo un poco es alterar esa r o ese comportamiento y Y de esa forma cambia por completo lo lo que el resultado del algoritmo No si es como como los perros de presa

no que van que tú le das a oler la camiseta del preso se llaman depra por eso porque persiguen a los presos Y entonces el al perro de repente lo que le gusta es ese olor Y entonces va mirando y dice por aquí hay un gradiente de olor y va Buscando el gradiente de olor solo le has dicho te gusta este olor eso es lo que tú le has enseñado y le da el olor le gusta el olor y hace lo que haga falta salta una valla hace lo que haga falta para llegar donde el olor es máximo eso sería lo que hace este algoritmo esto sería el perro pero tú le das el olor o sea que tenemos un perro de presa exactamente tiene un perro de presa pero tienes que que saber definirle qué olores le gusta y cuáles

no le gusta o sea en este punto del donde está el perro hay tales olores y yo te tengo que dar una función de te gusta más o menos porque está más presente el olor que yo te he dado que en el caso del perro es muy sencillo Cuanto más olor de este detestes mejor no pero en un caso digamos más real Pues tú puedes tener que al perro le guste perseguir ese olor pero también le guste

no morirse de hambre y entonces si tiene que perseguir los 5 días llegará un momento en que tire más el hambre que Y entonces parará comerá porque si no O sea que le tienes que poner varios objetivos que se solapan y que en un momento dado uno puede más que el otro para que al final él haga lo que tenga que hacer para dentro de una semana haber encontrado al preso fugitivo justo estaba pensando en eso

no que es lo que estabas comentando de la piedra que cae por la ladera porque no prevé futuros a vemos desde una perspectiva más amplia podemos ampliar el tiempo en el que estamos mirando y tenemos más eh acciones en cuenta para poder maximizar ese futuro no que lo que comentas el perro que tiene que comer que tiene que eh hacer más o menos su vida

para poder dar con con ese lor y llegar su a su meta Sí cuando lo vuelves a pensar ahora como física tú lo que tienes que esos futuros que tú estás lanzando están dentro de un espacio donde hay un potencial eléctrico por ejemplo no y son partículas cargadas Y entonces sienten el potencial y se van a la zona de potencial más bajo no ahí tiene una rar que es el potencial y que se va a esa zona ahora y si yo tuviese dos Campos superpuestos uno de carga y otro de

no sé qué no sé hubieran dos tipos de carga entonces cada partícula tendría su diferente porción de carga y le interesaría más minimizar un un campo minimizar otro o sea te gustaría más correr o te gustaría más guardar la energía y irá buscando una zona donde corras pero sea eficiente no gastes mucha energía si ya se te está gastando la energía está llegando a cero Entonces ya te parará dejarás de correr y todo eso se Auto está solo tú solo le das una serie de objetivos que son funciones digamos de cero a uno por decir algo y le dices lo que quieres maximizar el producto de las funciones exacto y ya te vale amente lo que va haciendo es encontrar ese equilibrio exactamente tú puedes decir mi mi nivel de energía va cer0 a uno mi nivel de salud va cer0 a uno Eso son dos objetivos los multiplico y energía por salud Ese es mi objetivo Por ejemplo yo soy un ser muy simple yo

no tengo objetivo en la vida excepto comer dormir y no morirme no eso sería ahí estaría definiendo su personalidad Pero él solamente hace eso ese cálculo y ahora utiliza a este perro de presa para buscar Qué acciones tiene que hacer para que esa multiplicación de lo máximo posible si está bajo de energía como está a punto de multiplicar por cero se convierte en su prioridad Porque si de repente come y sube al 50% buah todo sube pero si corre un poco más y sube o cuida un poco su salud dice a ver si te estás quedando sin pila

no cuides tu salud da un salto llega al enchufe enchufate robot no y el propio robot llega a esas conclusiones no pensando ni razonando Simplemente porque Busca maximizar una función que tú como humano le ha escrito es lo que decía del problema de quién escribe la función objetivo no pero si la función objetivo está bien hecha el perro no suelta presa o sea es superinteligente pero en este en este estadio

no tiene agencia no puede decidir qué le gusta hacer No qué va a hacer eso sí qué le gusta hacer porque al final yo no quiero decirte a ti Oye ve al frigorífico tira de la palanca Ábrelo V la mano no digo trame una cerveza yo te quiero dar órdenes de Oye no te gustaría traerme una cerveza que me pondría muy feliz y yo con eso planto esa semilla Y tú como perro de presa vas a hacer todas las acciones necesarias para ir y venir es objetivo claro v v que

no te queda energía y te vas a pagar pues te vas a enchufar primero no exactamente sería como el algoritmo de la romba pero generalizado donde una de las cosas que quieres hacer es limpiar y otra de las cosas que quieres hacer es mantener tu energía alta pero eso generalizado para poder tener 20 diferentes objetivos y que te los combine todos y que optimice la combinación de los 20 no eso fue un poco el trabajo después de llegar a lo primero que habéis visto es Cómo hago eso otro no y bueno la cosa ahí se complicó mucho eh seale que intent estaba preguntando algo pero no

---

## 9. Demo — l'evoluzione del cochecito e le prime emergenze

> **Tesi:** Sergio mostra video del primo cochecito: triangolino con sterzo binario, 100 futuri possibili rossi/azzurri, distanza media → decisione. Aggiunta del rozamiento, di obstacoli, di trampe. Poi il salto a punteggio per *distanza recorrida* invece che semplice conteggio: l'agente diventa veloce. I tre agenti con tre reward diverse mostrano emergenze radicalmente differenti.

Ah perdón mi micro se bajó y si este tipo de comportamientos eh que buscan maximizar la reward no sería como tal un razonamiento qué entraría en esta categoría de razonamiento Bueno si fuera solo maximizar una reward y tú mirase todas actuaciones y a qué punto te llevan Y qué reward tienen sería una optimización corriente y moliente Y si tú utilizas por ejemplo Monte Carlos TR Search ese algoritmo también es de planning y va intentando maximizar el objetivo tuyo que es la rivar aquí eso cambia tú

no quieres llegar al punto de máxima máximo reward tú quieres encontrar un sitio de Máximo reward y máxima diversidad O sea que si hay una zona no tan alta como un pico pero mucho más ancha lo va a preferir al pico es diferente o sea si la inteligencia no quiere alcanzar el máximo de dinero aunque se muera eso no la optimización normalmente está mal planteada en ese sentido Porque todos los sistemas de reformer lening intentan maximizar esa ruar futura pero

no se preocupan de el número de posibles futuros que vaya a tener igual si tú dices Mira puedes conseguir 100 de puntuación si hace esto pero solo hay esa opción si algo sale mal está muerto y tiene otro cam tu objetivo pero te destruyes por el camino con lo cual no tiene mucho sentido no Claro tú solo tienes dos posibles futuros o ganas 100 o mueres en el sentido de la entropía Eso es malo si tú tienes una zona donde digamos vale No hay 100 hay 50 60 70 pero V a montón hay muchos sitios donde ir hay

no hay problema No hay abundancia pero no está ese pico entemente eso es mucho mejor para cualquier humano y para cualquier [ __ ] no pero no para los algoritmos normales de de optimización que van digamos a por el dinero y eso no es lo inteligente lo inteligente el Montecarlo trer este tipo de algoritmo es uno de los algoritmo de búsquedas que tenemos e que se utilizan para tomar estas decisiones estratégicas pero son de los más eficientes que que tenemos Por así decirlo

no en base a búsqueda de e este reware con información incompleta porque tienes que ver distintos caminos y demás Entonces qué Qué diferencia habría con la aplicación de este algoritmo que que tú planteas sobre todo a nivel de complejidad de de cálculo básicamente los dos algoritmos están resolviendo

el mismo problema vale mirar a futuro qué va a pasar qué va a pasar qué va a pasar y decidir entonces a presente cuál es tu mejor opción

el TR Search intenta Buscar el futuro de máxima reward y si ese futuro empieza tú dando un pasito hacia la izquierda pues te toca dar un pasito hacia la izquierda o sea es lo mismo que hace el otro algoritmo lo hace de manera totalmente diferente basándose en cosas diferentes y con un algoritmo totalmente diferente Pero el resultado es el mismo de las tres iones que tiene te doy una puntuación para cada una y tú ya pues hacerlo evidente que cogerla de la más alta

no la diferencia sería que por un lado el el triarch el va solo mirando un futuro a la vez O sea que va haciendo un caminito vuelve atrás hace otro es como que va dibujando el árbol pero con un solo lápiz no hace un vuelve atrás y Elige a qué punto vuelve atrás y digamos vas dibujando un árbol con un solo lápiz y dibujando digamos en serie eso simplifica digamos que la gente lo vea diga tiene sentido Pero esto es muy muy muy muy poco óptimo lo óptimo sería dibujar 100 ramas a la vez y que se vayan entrelazando y que entre ellas se vean vean las distancias vean la entropía O sea Si pasas a dibujo 100 ramas simultáneamente interactúan entre ellas conforme crecen de repente tu algoritmo tiene acceso a mucha más inform ación y puede optimizarse muchísimo Entonces

el mismo problema

Por ejemplo uno de los problemas que estuvimos resolviendo era el clásico de los juegos de atari vale en los juegos de atari es justo lo que estamos hablando tiene no sé si eran ocho botones con lo cual tú tienes o c y 81 que tú puedes pulsar no pulsar no habían cosas o sea era muy sencillo a nivel de acciones tenía ocho bueno todas las combinaciones y ahí tú podías por supuesto intentar resolverlo ando un simulador de atari con lo cual

no tienes que construir el Word model y utiliz un sistema de planning que sería por ejemplo el monte Carl research Y entonces encontramos un paper que hacía todo ese estudio y lo comparamos con lo que hacía el nuestro que también lo teníamos ya adaptado que claro no iba mirando en futuro sino que jugaba 100 partidas a la vez y las partidas malas clonaba a la buenas era como un algoritmo digamos más genético no

y nos encontramos por ejemplo Que que

---

## 10. FMC vs MCTS — il salto di efficienza

> **Tesi:** Sui giochi Atari: MCTS UCT richiede ~150,000 letture del simulatore per decisione. FMC ne richiede ~35 — **3 ordini di grandezza meno**. Su problemi di robotica continua (figura 3D che cammina) MCTS è di fatto impossibile computazionalmente; FMC li risolve con ~5,000 futuri per decisione. Robustezza al rumore: degrada linearmente, autocorregge ad ogni passo.

el Montecarlo research utilizaba para resolver cierto los juegos creo que era del orden de 150,000 lecturas del entorno o sea le preguntaba 150,000 veces Qué pasará así a su World model para tomar una decisión dentro del juego y el nuestro igual eran 35 ostras

Sí sí era una diferencia importante totalmente incomparable o sea era una cosa que cuando realmente quería hacer una comparación de decir Bueno voy a poner algo que al mío le cueste Monte Carlos triser era totalmente incapaz de hacerlo o sea solamente usando planning nosotros conseguimos que un modelo de estos 3D de un muñequito que anda que ahí lo que tiene son la fuerza en cada una de de de la de las uniones el torque que le va a aplicar y ú simplemente decide si subo o bajo el torque en cada sitio si tú eso te lo planteas como un problema simplemente de planning es decir mi reward es la altura de la cabeza yo quiero que esté de pie

Y entonces ahora decide en cada momento qué torque aplique a cada una de de de las uniones eso hacerlo con Monte Carl research nadie lo ha hecho Simplemente porque no hay computación en el mundo para eso claro es muy costoso nosotros lo hacíamos con 5000 futuro por decisión 5000 re por decisión o sea a nivel de lo bien que funciona y lo eficiente que es le da vamos está está está en otro nivel totalmente de cualquier otra cosa que exista eh Porque el Montecarlo triser sería el tope de gama ahora mismo o sea que a ese nivel básicamente yo Considero que se ha resuelto El problema del plan ineficiente No yo no no concibo manera más eficiente de hacerlo

sobre este punto del modelo del mundo has hecho alguna especie de experimentos o ablations de de cómo funcionaría el planning con un modelo del mundo defectuoso ligeramente incompleto o algo así sí sí claro cuando utiliz un algoritmo cuando utiliz un simulador claro es todo determinista y digamos está haciendo trampa Pero tú puedes hacer que cuando tú lees el el estado puedes sumarle un ruido Entonces yo creo que mi coche está aquí pero realmente está 10 cm a la izquierda o a la derecha y también puedes sumarle ruido a tu decisión es decir yo he decidido mover el volante hacia la izquierda 3 grados igual se mueven dos pero como este sistema en cada decisión Mira Su futuro a partir de su posición actual si hay un una pequeña desviación entre lo que él pensaba que iba a ocurrir o lo que él pensaba que iba a hacer y lo que ha hecho se autocorrige automáticamente O sea que sí que se degrada el algoritmo pero de una manera digamos lineal con el ruido que le mete y muy baja o sea lo normal Por ejemplo si tú tienes tienes un coche paso vuelva a calcular otra vez aunque le hayas metido ruido a su decisión o su posición inicial como va a calcular el siguiente paso Desde esa posición volverá a corregirse

no Sí bueno Y aparte también cuando tú vas generando los futuros esos futuros van acumulando errores porque la esa predicción entonces claro tu futuro es un futuro que igual luego en la práctica No se puede realizar porque las cosas más lejanas no están muy no son muy muy posibles claro dice en lugar de acelerar a 100 por hora que es lo que yoed o sea yo puedo acelerar tanto si realmente le digo acelera a tope y acelera al mínimo y llega un momento en que tú ya Hay ciertos futuros que

no vas a poder alcanzar y que tú creías que sí eso no tiene importancia o sea eso desaparece digamos se cancela normalmente Y sí tú notas una diferencia baja el rendimiento por cualquier métrica pero totalmente lo esperable cuando es ruidoso O sea no no no no se afecta el sistema no supone que la función la riguar el estado nada sea derivable ni continuo ni nada Eso es lo bueno como un sistema discreto hay 100 puntos que se están moviendo y y

no se supone nada sobre la continuidad de nada simplemente tiene 100 puntos donde tiene 100 recompensas 100 distancias entre ellos y te da igual que tu mundo sea continuo discreto lineal caótico todoo eso te da igual el algoritmo es en ese sentido es totalmente genérico sobre todo en la rar le puede meter un ruido terrible si de media una zona está mejor él va a ir a esa zona da igual que el ruido sea terrible Ah ahí sí que no no no hay problema

vale genial podemos seguir viendo cómo evolucionó el algoritmo Sí claro claro bueno esta fue la prueba donde ya le metí la primera recompensa por distancia recorrida no

Y entonces aquí comparaba uno que no tiene esa recompensa el amarillo digamos y lo el negro y el blanco sí tienen esa recompensa uno la distancia recorrida y otro es la distancia al cuadrado y bueno lo que se ve claramente Es que le ha le ha dado vidilla no de repente quiere ir lo más rápido posible y Y ves que sí que le has cambiado el comportamiento el de abajo sería el sentido común voy andando

no me meto en follones y el de arriba ya es tengo un objetivo que quiero cumplir y encima le podía subir la importancia del objetivo porque al final esos objetivos son números que se van multiplicando pero cada número Yo luego jugaba a elevarlo a algo si lo elevas a un no haces nada Pero conforme lo elevas a más le da más importancia a un objetivo y podía modificar la importancia del objetivo y en tiempo real y modificando lo que le importaba una cosa y otra no

vale Y en este que estamos viendo ahora en este fui un poco más lejos y dije Bueno voy a hacer tres y a cada uno le voy a poner una función objetivo totalmente diferente el naranja cuando toca Una un círculo naranja lo absorbe digamos y es como si fuera miel no entonces sube su energía Entonces siempre que que que está recibiendo naranja va subiendo su energía y entonces lo que él quiere suo energía a ver qué hacía

no el blanco lo que quiere es ir lo más rápido posible sería el original y a este de aquí que tiene una línea lo que dice fue que quería eh maximizar su velocidad radial o sea A qué velocidad se mueve el radio verde no O sea le daba igual su velocidad personal sino que quería ir lo más rápido posible simplemente para ver cómo cambiaba el comportamiento

no para ver qué pasaba O sea que esto es lo que hablamos al principio de que modificando un poco es redw se van adaptando sus comportamientos a esos objetivos si fijaron el naranja Cómo va va su bola literalmente literalmente va su bola aquí solo tenía un reward por persona digamos elegía una función aún no había aprendido a mezclar funciones pero básicamente esto aquí ya cuando ya empieza a mezclar y va haciendo ese cóctel que ya te permite crear un [ __ ] soltarlo y Volver al día siguiente y que siga vivo Porque si ha llevado ha llevado cuidado de

no chocarse ha llevado cuidado de comer ha estado haciendo No sé qué Y si encima le has dicho Oye y cuando tenga un rato me V escribiendo una novela cuando vuel te escrito una novela No aquí ya empezaba a ser un comportamiento que tú veas más biológico más que superinteligente lo veías biológico No aquí por quieren [ __ ] aquí mezclé no quieren [ __ ] las bolitas esa de comida y

no quieren chocarse porque les baja Entonces simplemente están esto sería un animal básico que quiere comer y no chocarse Claro no Entonces está como teniendo ese comportamiento que hablamos antes donde va un poco equilibrando los objetivos no exactamente pero si te fijas se había muerto uno eso Porque este algoritmo aún no era fractal como yo llamos

no aquí seguía creando 100 futuros por cada una de mis acciones posibles y comparando el número de futuros diferentes Claro porque lacia de este algoritmo está limitado un poco a lo que es capaz de proyectar a futuro no es claro hasta ahora lo que lo que habéis visto tenía ese problema que yo estoy proyectando líneas a futuro eso es muy muy muy poco eficiente hay un vídeo que compara con y sin la siguiente la siguiente mejora digamos que que le añadí que si si quieres te explico un poco la mejora primero a nivel teórico

no sé si a nivel teórico te suena a malo o a buena cosa al principio teníamos ese cono que queríamos calcular su entropía y que no había manera no es muy complicado Pero cuando empieza a investigar a qué se refiere aumentar la entropía lo máximo posible llega a que hay unos teoremas que te dicen que la entropía se aumenta cuando la distribución de recompensa es totalmente igual a la distribución de tus partículas digamos O sea que en un gas el potencial es proporcional a la densidad de partículas que tiene que haber ahí vale o sea que es lo que comentaba antes en la zona de ese cono donde hay más rar va a haber más densidad de caminos que pasen por ahí es digamos un poco de cajón

no pero eso técnicamente lo que quiere si tenemos unos agentes dentro de ese cono o una población en sitios donde haya una densidad de entropía mayor habrá un número mayor de agentes no densidad de entropía lo que habría era densidad de ruar o sea es como el potencial en la zona donde tú estás tú tienes un montón de trabajadores y lo sueltas hay en el campo

no y en el campo hay oro y ellos van Buscando oro vale Y entonces ellos lo que quieren es que en el en el suelo donde están ellos donde van a excavar haya más pepitas por metro cuadrado eso es lo que quieren optimizar Per la densidad de la ruar no de la densidad de AR ruar Y entonces lo que te viene a decir es que si tú miras en un en un territorio y haces subdivisiones en la zona donde haya más densidad de oro Habrá más densidad de buscadores de oro tiene toda la lógica tiene toda la lógica no en termodinámica eso

---

## 11. Il salto fractal — da linee aleatorie a sciami evolutivi

> **Tesi:** Il limite del primo algoritmo: simulare cammini lineari indipendenti spreca. L'analogia dei **mineros**: se tutti scavano nella stessa zona, devono dividersi l'oro. La distribuzione ottimale (Boltzmann) ha *densità walker proporzionale alla densità di reward*. L'algoritmo diventa **evolutivo**: i walker più poveri clonano i più ricchi, ma anche la **distanza** conta (anti-collapse). È il salto da O(N) lineare a O(N) parallelo.

se llama la distribución de wolfman que te dice el gas va a atender después de hacer todo lo que tenga que hacer se va a buscar su situación de equilibrio donde la densidad del gas sea digamos paralela a la densidad DEA al valor en este caso sería el potencial potencial eléctrico o lo que sea y física como el potencial se ve ir al revés pues se está minimizando igual de maximizando Pero bueno en el caso del Oro yo lo veo más claro más oro más gente claro cuando tú eso lo ves a nivel entropía lo que estás diciendo es que esas dos distribuciones tienen que ser proporcionales o sea tien que ser iguales con lo cual su entropía Cruzada tiene que ser máxima O sea que tú realmente todo el cono lo que te está diciendo es que tienes que maximizar la entropía Cruzada entre la distribución de rar y la distribución de tu Walker lo cual tampoco es que te haya sacado de pobre porque eso sigue siendo complicado

no pero ahora cuando tú lo piensas desde el punto de vista de ese Walker de ese punto de ese ser imaginario que está del minero para el minero es muy sencillo cantidad de oro dividido por cantidad de personas a más oro mejor a más a más personas peor claro tienen que compartir claro Entonces si tú quieres calcular Cuál es la distribución que hace que todo el mundo esté contento en el sentido de que hay menos oro pero es que hay menos gente esa distribución sería bman y sigue siendo igual de complicada de calcular pero cuando lo piensas desde el punto de vista de uno de esos mineros es muy sencillo tú le puedes decir Oye cómo de buena es tu situación y dirá 50 G de oro por metro cúbico dividido por 30 personas que hay en mi alrededor 50 partido por 30 así de contento Estoy otro dirá 200 G de oro y 10 personas aquí noos estamos forrando el mío es 200 partido por 10 y entonces claro el el otro minero pues lo mirará hablará con el móvil por ahí le contará Uy aquí en Estados Unidos va todo perfecto

no y dirá buah yo me quiero ir ahí sea porque estás comparando como de justo digamos el reparto que es exactamente lo que te dice la la la distribución de bolman O sea que tú simplemente cuando te bajas al nivel de las personitas de de las particulas Ellas quieren una cosa muy simple más y repartir con menos una cosa dividida por otra si tú ahora quieres resolverlo

no usando la fórmula sino haciendo un algoritmo evolutivo donde los 10 más pobretones que tienen menos oro por persona se copian a los 10 más ricachones y se modifican con un algoritmo genético Vanilla digamos lo tendrías resuelto podrías calcular claro ahí podrías calcularlo perfectamente con un algoritmo genético normal y corriente bueno evolutivo la gracia es que todavía se puede hacer mucho mejor que un algoritmo evolutivo O sea que básicamente lo que vaya a ver es un algoritmo evolutivo evolucionado de digamos de segunda generación que utiliza algunos trucos adicionales usando el concepto de entropía que simplemente quiero que además los puntos estén separados

no solamente quiero el 10% mejor quiero que el 10% mejor y que esté en zona más dispersa cuando tú metas metes esa segunda métrica de decir no voy a matar al 10% de peor rard y lo voy a llevar al sitio de 10% de mayor rard sino de peor situación en en su entorno es decir gramos de oro dividido entre personas cuando cambia a eso y dice ahora los 10 más desgraciados En ese sentido van a clonar a los 10 más Afortunados En ese sentido teniendo en cuenta que entre ellos compiten de repente el algoritmo genético pasa a funcionar muchísimo mejor y ese salto que de repente se ve en la en en Cómo de bien funciona el truco es tan simple como eso la distancia importa tanto como el dinero sí qui lo hubiese dicho Eh sí sí sí sí y en el fondo es como que si tuviera tuvieras dos objetivos en la vida el oro es uno y el otro es digamos la soledad estar alejado que eso lo ves también en la vida digamos de de de de de un investigador de de de de tema de Inteligencia artificial que dice vaya los Transformers está todo el mundo con los Transformers sí que va muy bien pero hay mucha gente investigá olo y ahora ha salido aquí el mamba que resulta que todavía

no se sabe si va bien o mal pero hay cuatro gatos investigá olo Igual me interesa dedicarle un tiempo al mamba a ver si consigo que vaya tan bien como el otro y de repente soy el que lo ha hecho florecer tengo simil que hemos puesto hace un momento de los mineros no sí sí sí hay un montón de Mineros ahí con su recompensa claro por su población y hay unos poquitos dice Pues lo mismo allí

Claro ese principio que cuando te lo cuentan di Bueno eso es cotidiano eso todo el mundo lo sabe no vale Pero si ahora lo convierte en un algoritmo genético bueno evolutivo donde simplemente tiene en cuenta las dos cosas de repente el algoritmo evolutivo explota en eficiencia de una manera que nadie se lo esperaría si encima consigue Que el algoritmo sea on O sea que

no cueste mucho calcularlo que puedas tener miles de walkers y no tengas que calcular todas las distancia entre ellos Si lo consigues simplificar a nivel de puro algoritmo Y ser muy eficiente y lo consigues vectorizar y meterlo en una gpu y no sé qué pues tú te puedes plantear mirar a futuro largo con 50.000 puntos como una especie de nube de futuro que te puede pasar una nube de gas que evoluciona seguiendo este algoritmo Tan sencillo evolutivo Y tal Pero que en el fondo sabes que está optimizando la entropía futuro pero

no te has tenido que ensuciar las manos con eso de entropía futuro el logaritmo de no sé qu no te has tenido que forma estás aumentando lo que venimos un poco hablando de esa inteligencia no estamos haciendo Cuanto más futuros o más distanci futuros podamos medir más inteligente o mejor resultado nos va a dar el algoritmo no exactamente si el algoritmo estuviese bien hecho darle más segundos para pensar a futuro le haría más inteligente esto un poco lo que estamos viendo ahora en estos modelos razonador de llm con o3 por ejemplo a la cabeza que ellos indican que Cuanto más tiempo de cómputo gastan en su respuesta más inteligente o más razonada es la respuesta

Sí con la diferencia de que ellos realmente lo que calculan es un solo camino Ellos están calculando el camino más probable lo único que ese camino digamos que primero se dan un paseito hacen los token de razonamiento se pasean por la zona miran a ver los problemas que va a tener y cuando ya lo han visto todo y con todo ese texto dentro de su contexto ahora se ponen a responder o sea que sí volvemos un poco al caso del Montecarlo eh exactamente exactamente las redes neuronales están avanzando mucho cada vez razonan pero van a chocarse con un muro que

no es el que la gente se imagina el sistema va a poder seguir mejorando Pero si siempre va siendo lineal va siendo creando solo un camino se va a chocar con un muro de que siguiendo un solo camino encontrar el camino digamos genial es muy difícil porque sigue añadiendo token al tuntun uno a uno y eso nunca te va a llegar te va a llegar a ciertos sitios pero no te va a llegar no va a llegar a los sitios más interesantes digamos no si tú quisieras digamos Imagínate que tú tienes

---

## 12. Il problema del labirinto — bengala vs láser

> **Tesi:** Algoritmi lineari (token-by-token) sono come **un puntatore laser**: trovano ciò che è in linea retta. FMC è come **lanciare una bengala in un labirinto** — il gas riempie lo spazio, una particella trova l'uscita. Per problemi con spazio vasto e cammini stretti (i labirinti complessi della cognizione), solo approcci a sciame funzionano.

una habitación muy compleja y tú quieres analizarla y utiliz un rayo láser y vas disparando a diferentes sitios No eso sería aproximadamente lo que hacemos ahora y lo que hace este algoritmo es abrir una bengala y que se llene todo de gas y ahora de repente una partícula ahí le dice e he encontrado un agujero por aquí llegamos a un sitio y to tú dices eso con el láser

no lo V a encontrar nunca no entonces hay muchos problemas que son de ese estilo que tiene mucho espacio pero solo hay un caminito muy estrecho que te lleva al éxito y es muy difícil de encontrar porque el espacio es laberíntico no entonces el algoritmo entre comillas tiene que saber resolver el laberinto internamente para llegar a encontrar el sitio que si

no si no lo ve no sabe que tiene que ir por ahí no Entonces se complica todo y por eso el G es muy bueno porque todo el gas lo suelta en un laberinto y lo explora naturalmente no Y si encima haces que el laberinto tenga digamos atracción un buen olor en la en el punto de salida y le pone olfato a esas partículas encima ya van al sitio es que van al sitio es que lo resuelven inmediatamente de esa manera

no podemos verlo por [Risas] favor en los que están interactuando más de una gente dentro la simulación Cómo estás calculando O cuál está calculando primero los movimientos del otro endiente y cada gente es independiente pero como van calculando los futuros todo simultáneamente en paralelo cada gente puede ver lo que está viendo el otro tampoco es que te dé excesiva información porque todos los caminos son aleatorio pero realmente los caminos son comunes quiero decir aquí ahora mismo en esos vídeos tú

no estás viendo cinco inteligencias cada una controlando un un bueno perdón en el que está viviendo antes de los coches Sí pero realmente tú lo puedes plantear de otra manera decir yo tengo un robot que tiene cinco dedos que son mis cinco agentes que son mis cinco helicópteros que tengo en el aire pero el estado es la suma de los estados de los cinco entonces para mí es un solo objeto con muchas coordenadas y que se puede mover de una manera o de otra pero para mí a nivel abstracto es solo es solo un objeto y yo tomo decisiones sobre ese objeto y cuando yo ahora mis decisiones se las comunica a cada uno cada uno toma la parte que le esa y funcionan como un solo un solo una sola entidad hay solo una inteligencia controlándolo a todas si tú quisieras tener inteligencia digamos independiente una de otra que

no se comunican que no ven los futuros del otro sí Entonces tu Word model debería contenerlo a él y tú decir yo estoy simulando lo que va a pasar allí veo a un compañero que como va en esa dirección mi Word model me dice que se me va a cruzar pero ya es cuestión en ese momento ya todo eso Word model ya digamos el algoritmo dice espero que haya aprendido bien lo que pasa cuando te cruzas con esta gente porque si

no no nos chocaríamos no O sea que tú ahí en ese en ese sentido tú dependes de tu World model que es lo mismo que ocurre cuando estás jugando ajedrez una máquina contra otra tú estás prediciendo lo que el otro jugaría si fueras tú aún no lo has visto tú no sabes si eso es lo que él está haciendo Entonces tú en tu mente juegas las dos partidas no una eres tú y la otra es tu entorno entonces tú dices yo voy a hacer esto Qué creo que hará el entorno en ese caso y tu simulador de aje dice el entorno va a mover el alfil aquí que es tu oponente O sea que siempre está lo tuyo que tú decides y todo lo demás el entorno tienes algún experimento en el que en el que estos agentes estén como de manera independiente interactuando sí tengo independiente bueno independiente lo que habéis visto hasta ahora es independiente cada uno va a su historia

no eh pero si os fijáis en los vídeos que habéis visto de los coches entre ellos no se chocan son como tres vídeos superpuestos no cada uno vive en su mundo no ahí no hay no hay no conviven No es simplemente comparar qu hace digamos un agente con un algoritmo con otro algoritmo con otro algoritmo si vamos más adelante vamos a ir directamente al que tú me estás preguntando a ver Bueno este ya sería un ejemplo donde se nota mucho si es inteligente o

no Porque el sistema es muy complejo aquí ya en lugar de cochecito y tal pasé a simular un cohe Tito O sea que esto sería dos dimensiones pero digamos arriba y abajo y este cohete tiene dos grados de libertad que sería hecho más fuego hecho menos fuego es a la fuerza y luego lo puedes girar en un sentido o en otro para rotar tienes do grados de libertad pero para complicarlo

---

## 13. Il razzo con uncino — sistemi caotici e cooperazione multi-agente

> **Tesi:** Razzo 2D con gancio elastico: attrattore caotico, molto difficile da predire. FMC lo controlla bene anche mostrando comportamenti emergenti (lancia il gancio da una direzione opposta). Tre razzi cooperanti per spostare un masso pesante: **costruiscono un fionda** spontaneamente — non perché pianificato, ma perché guardando avanti era l'unica opzione viable. Una sola intelligenza, tre dita.

le he puesto un gancho con una goma elástica con lo cual ojo elástica quiere decir que yo estoy manejando el cohete el gancho detrás con su masa va a estar po yendo para un sitio y para otro es muy difícil de predecir es un atractor caótico en física por eso se lo puse no porque porque quería quería complicándose función de riguar eh yo no he sido el culpable entonces la rigual que le dice fue eh el gancho quieres que esté lo más cerca posible de un asteroide entonces detecta el asteroide más cercano y intenta minimizar la distancia gancho asteroide cuando lo lo atrapa entonces cambio la función de rwar y quieres que tu gancho esté lo más cerca posible de la zona verde y entonces esa combinación de riguar hace que quiera tomar tomar asteroides subir ir con el gancho que es muy difícil y llevarlo hasta dentro de esta digamos hasta la canasta

no en ese momento el gancho se suelta y luego a continuación si puede lo vuelve a [ __ ] lo vuelve a subir lo vuelve a poner no este fue el digamos el caso digamos que a nivel físico es más difícil de hacer y bueno lo hace muy bien Ahora veréis aquí por ejemplo fijaros que claro él está aprendiendo a volar solo y bueno hace lo que se espera de él no V lo atrapa lo sube por muy difícil que sea tema pero lo curioso es lo que pasa que uno

no se espera que que es lo que vaya a ver ahora mismo cuando aquí se le escapa mirar por ejemplo lo que hace ahora es tira el gancho para que vaya coja el rojo y lo suba sin esfuerzo y de repente de algoritmo no no te oigo que decía que ha conseguido estirar La goma arriba para poder enganchar a la pieza roja y subirla no sí el algoritmo sigue siendo el mismo igual de Simple pero lo que ocurre es que aparece en comportamiento digamos claramente inteligentes emergentes de esa maximiza la entropía que realmente fue lo que más me sorprendió eh o sea igual que por ejemplo cuando char gpt empezó a decir cosas coherentes y empezó a hacerlo muy bien me sorprendió mucho en este caso este fue el momento ese que dije vaya supera las expectativas

no y el siguiente experimento fue el que tú me dices César Que y si ahora le pongo una piedra muy grande de manera que no pueda con ella pero pongo tres agentes que están colaborando O sea que solo hay una inteligencia Pero hay tres agentes colaborando qué ocurriría qué emerger yaa de eso sería también inteligente pero a la misma vez eh colaboraría uno con otro y la respuesta Ya te la imaginas pero otra cosa es verlo solo hay una inteligencia eh o sea que los tres van gobernar por la misma y fijaros lo biológico los robóticos que comentabas

no sí como si cada uno fuera un dedo de un solo robot no O sea que solo hay un estado solo hay una entropía solo hay una cosa y lo curioso es lo orgánico que es no Cómo parece que están ahí como si fuera un Enjambre se acercan uno lo coge cuando ve que ya está tirando dice Bueno ahora ya voy a ayudarle que puedo Fíjate cómo se van sumando uno otro bu y entre los tres logran subirlo arriba

no Uno no puede no puede Ni subirlo uno no puede Ni subirlo y entre dos si os fijáis entre dos tampoco pueden o sea pueden subirlo pero no para o sea le cuesta mucho que han conseguido subirlo pero no en su sitio porque eso es más difícil no lo consiguen y esto esto que va a pasar ahora Para mí fue lo que me dio miedito dije madre mía mirar el rojo mirar el rojo lo que hace Está en dirección contraria

no está haciendo un tirachina ahora lo suelta y han digamos han construido una herramienta un tirachina entre los tres no porque era la única manera que tenían pero no es que lo hayan pensado no es que lo hayan planeado es que a futuro es la única opción que han visto es la mejor que han visto Mirando a futuro con ese algoritmo fractal el truco está en lo que

no hemos dicho y es que ya no son líneas si fueran líneas no podrían ver ese futuro no tendrían esa capacidad sería ese agujero que solo ves y sueltas un gas porque las probabilidades de llegar a esa conclusión son muy pequeñitas es un poco lo que comento de lo que yo creo que le va a ir pasando a las redes neuronales en el sentido de que tienen la limitación de que ven un futuro trazan una línea trazan una frase solamente en ese nivel siempre van a tener los problemas que que vaya a ver aquí que tuve yo en su momento Y es que en mi caso los movimientos eran totalmente aleatorios

no es que yo eligiese el token más probable aquí elegía token turn no pero el problema es parecido si yo elijo movimientos aleatorios por muchos movimientos que yo simule aleatoriamente muy difícil que yo encuentre un camino que salga de aquí yo no puedo avanzar mi visión con ese algoritmo no puede ver más allá de estos metros que está viendo y tú le puedes subir el número de puntitos que está calculando lo puedes multiplicar por 1000 y se y va a aumentar un 10% la longitud y ya

no pueden más tiene un tope donde hasta donde puede de ver ese era el problema que tenía el algoritmo original el que hice en un día y que bueno Por más que logré tirarlo el problema que tenía era ese de ahí ese problema hacía que no pudiera no pudiera funcionar Le faltaba la parte que digo evolutiva donde esos mismos futuros ahora ya los los trato como individuos y les permito hacer cosas queé es lo que V a ver aquí ahora ahora explicaré un poco cómo funciona pero lo que ahora estáis viendo el algoritmo ya el el que utiliza esa es ese evolutivo ahora ya le puedo poner todos los segundos que yo quiera y el va a seguir mirando hacia el futuro hacia el futuro y entonces se eliminó Esa esa ese problema del tiempo y ese problema de que el árbol se bifurca tanto que ya

no te cabe en la memoria explota No aquí tú le fijas quiero 1000 futuros pero quiero que los 1000 futuros estén distribuidos de la misma manera que estaría un gas real expandiéndose en este espacio y entonces esos 1000 futuros digamos lo utilizas de manera óptima con 1000 haces lo más que se pueda hacer con 1000 no y aquí ve aquí Vais a ver uno de esos procesos de pensamiento que aquí pasan como un flash visto en cámara lenta Este es el algoritmo evolutivo esos puntitos son la población y van evolucionando van cambiando No aleatoriamente aquí van cambiando en el tiempo y entre ellos se van clonando se van copiando en función de que tú vas más rápido Yo estoy en una zona llena de otros puntitos usando ese algoritmo Tan sencillo consigue trazar un camino porque situar cualquiera de esos puntos le trazas marcha atrás toda la historia por dónde ha llegado hasta ahí tendría tus 100 caminos que logran salir

no tendría un árbol de caminos que empiezan donde tú estás y salen totalmente fuera pero que pueden terminar el laberinto no sí sí lo terminan completamente

Bueno fijaros cuando llega aquí como es como un gas se expande por la habitación no O sea siempre se está comportando como un gas como la onda de choca de un de choque de un gas digamos no y qué os quería comentar también antes de dejar todo este tema Bueno si queréis seguimos viendo el vídeo tiene más partes y más cosas Pero básicamente podéis ver por ejemplo pues cómo jugaba a juegos de atari cosas de ese estilo en

---

## 14. Atari, RAM-as-state, e l'agnosticismo dello stato

> **Tesi:** Lavorando sui giochi Atari Sergio scopre che usare il **dump della RAM** come stato funziona 10× meglio dell'immagine pixel — anche se 'non vede' ciò che succede. L'algoritmo è completamente agnostico al significato dello stato: vuole solo un vettore predicibile e una distanza calcolabile. Il vero lavoro umano è **definire la reward** sul problema specifico.

los juegos de atari es curioso Te lo comenté personalmente a ti Alberto y es que tú Aquí Estás usando el estado pero cuando tú luego te enfrentas a un problema real uno tu primera pregunta es queé es mi estado Cuál es el estado de esto por ejemplo si tú estás hablando de una llm tu estado Pues sería el promt más el texto que ya has generado ese sería tu estado interno digamos de la maquinita que está pensando el siguiente pront Pero cuando te planteas una máquina de atari donde tú

no sabes las coordenadas del protagonista son píxeles en una pantalla no hay coordenadas no hay ha velocidades no hay nada físico Simplemente hay un bmp que te da tu pantalla Bueno pues el bmp se podía usar como estado se convierte en un vector y es es mi estado y yo puedo comparar bmp y funcionaba perfectamente pero seguía teniendo problemas en el sentido de que es un estado muy raro que de repente cuando Gan empieza la pantalla a parpadear y tu estado da salto de un punto muy lejano a otro entonces claro para calcular distancia era un estado malo y aún así funcionaba y hacía todas las partidas muy bien y se nos ocurrió decir bueno a ver nosotros tenemos el volcado de la ram de ese simulador sabemos Ahora mismo cómo está toda su Ram su bits su variable interna tenemos esa información que es raw

Sí pero es que un bmp puesto como un vector eso es raw también eso no se entiende Pero es un estado y entonces probamos a usar como estado la ram el volcado de la ram y funcionaba 10 veces mejor tú decías madre mía Pero si no está viendo lo que pasa pero claro es que no ve no se dedica a ver Busca entropía no entonces

tan alguien que cuesta a veces hacerse la idea de lo que está pasando dentro pero al final es tan sencillo como lo de los puntitos de gas no Simplemente que ese puntito de gas corresponde con el volcado de la ram de de un videojuego no Y entonces a veces lo que más cuesta es ver el paralelismo entre true problema y la la solución digamos teórica

está viendo la representación que sale por la pantalla lo que está haciendo es la representación del sistema en esa memoria RAM en algo en específico no que puede ser un cambiando o o algo parecido no esa variable que estás teniendo en cuenta exactamente como el algoritmo es totalmente agnóstico a lo que significa el estado simplemente un vector del cual solo necesita predecir el siguiente vector y calcular la distancia entre dos vectores lo que signifi el vector le da igual Entonces eso te da una una flexibilidad para poder adaptarlo casi a cualquier cosa pero siempre hay ese interface humano que dice esto lo voy a adaptar así y sobre todo le voy a poner este rigual ahí donde está donde tengo la espinita clavada

no tengo que hacer que la riguar se calcule sola se la invente que yo lo suelte y él diga vaya Gotita de miel yo creo que me van a gustar noio aquí hay un par de preguntas en el chatal si te las comento Eh Fato nos dice A qué se refiere con fractal Cómo se interconectan podrían explicar el algoritmo fractal Sí bueno

---

## 15. Cos'è 'fractal' — auto-somiglianza a tutte le scale

> **Tesi:** Risposta a una domanda dal chat: fractal = generalizzazione dell'albero. L'algoritmo ha la stessa struttura a tutte le scale temporali: dalla decisione microsecondica del corpo, al piano del giorno, alle scelte generazionali dell'umanità. Stessa equazione di clone + reward + distanza. Per questo si chiama Fractal AI.

fractal es un adjetivo que a alguna gente no le gusta no hay un poco en matemática un poco como decir conciencia no en Inteligencia artificial Ah fractal claro todos son fractales ya ya fractal es digamos la la la la generalización de un árbol tú tienes un árbol el árbol el típico árbol que se usa siempre porque lo podo imaginar se va bifurcado pero el árbol es el primer fractal que se descubrió Leonardo Da Vinci dijo un árbol es algo hecho de árboles más pequeño a cada rama de un árbol tú la cortas la pones en el suelo te alejas y si

no te has dado cuenta del truco dices Mira dos árboles uno grande y uno pequeño eso es un fractal en este sentido esto sería un fractal lo que está generando o es un árbol depende de la escala si tú te imaginas este camino que va siguiendo a una cierta escala voy pegando saltito de una cierta escala temporal tú tienes un árbol clásico ahora tú ese campo ese salto temporal lo podías hacer más pequeño y el número de de digamos de puntitos lo podías hacer más grande y si es cierto que en el límite el árbol se convertiría en un fractal como cualquier árbol eh o sea que esto

no no estoy no estoy inventando nada en la realidad fractales no existe porque no hay nada que se repita infinitamente tú puedes tener una coliflor que por cierto tenía una que iba a subir que tengo un romanesco que es un fractal perfecto pero se me ha olvidado abajo pero bueno Tú puedes tener una coliflor y tú cuando arrancas una rama de la coliflor y la miras de cerca es una coliflor Entonces eso es un fractal autosemejante a escala se dice o sea que en lo pequeñito y en lo grande básicamente es lo mismo esto por qué le llam yo fractal porque fijaros que hemos dicho que conforme va bajando el tiempo puedes llegar hasta la física entonces este simulador de coche Yo podría haber llegado al extremo de decir

no voy a hacer un simulador de física el simulador de física lo voy a hacer con un fractal que funcione en tiempos de plank bueno no habría llegado tan pequeño porque si no aún estaría simulando pero yo podría decir yo simulo lo que voy a hacer día a día digo hoy mi decisión es tal Y entonces con esa decisión paso el día y podría hacer mi árbol de qué cosa podía hacer durante el día y luego decir bueno

Y ahora qué hago durante el día ahora cojo otro fractal más chiquitito de cosas que voy a hacer la siguiente hora y podría imaginarte que a cada escala tienes una historia de pasos que vas dando día a día mes a mes o en esro sentido minuto a minuto segundo a segundo y en todo hay la misma escala o sea cuando tú vas a tomar una decisión de microsegundos cuando tú vas a decidir si vas por un lado de la mesa o por otro tú estás haciendo esto tú estás viendo que por este lado el el el pasillo es más estrecho igual luego

no puedo pasar porque hay algo por el medio este lado es más ancho y entonces instintivamente va por el lado más ancho porque tiene más posibles futuros digamos o se que es una cosa que hacemos a todas las escalas y cuando llega a la escala mínima pues la la naturaleza con su segunda ley de atónica también lo hace Entonces yo podría dibujar este este árbol cada vez a escalas más pequeñas sería autosemejante y podría llegar a una escala de plan en principio y hacia arriba podía decir la humanidad como S como de esto estos que están colaborando como los de los de de una mano la humanidad como tal también es un ser que toma decisiones generación tras generación y puede ver su árbol de que podría pasar en el futuro o sea hacia lo grande y hacia lo pequeño sigu encontrándote la misma estructura el mismo árbol que sigue las mismas leyes de voy clonando voy a O sea que el mismo algoritmo lo puede aplicar a todas las escalas de del universo digamos

no Entonces en ese sentido es por eso está en la palabra Aunque lo que yo presento aquí es una cierta cala realmente es un árbol no sé qué ver que en mi cabeza está todo eso Entonces yo aunque presente una parte yo me estoy imaginando todo y y para mí un fractal de libro pero luego la gente me lo discute y me dice hombre eso es un es un árbol digo bueno que un árbol qué un fractal sí a una escala determinada claro Sí aquí hay otra pregunta de a verel maestro que dice me queda la duda de Cómo buscar las semillas fractales que llevan a las recompensas podríamos hacerlo recursivo la búsqueda de los objetivos Yo veo que atención es todo lo que necesitas

Sí a ver el tema de cómo elegir lo las funciones de rar fijaros que los otros Pilares cada uno tenía una ley física detrás y más o menos Estaba la cosa Clara Cuál sería la tercera ley física que que que aplicaría para este tercer fractal eso le iba dando vueltas y mi intuición es la siguiente en física tú Imagínate que tú estás diseñando un Fórmula 1 y entonces tiene el simulador este de flujo del Cómo pasa el aire y haces todos tus cálculos y tú vas subiendo la aleta y va cambiando todo ese sistema funciona calculando flujo laminar solamente o sea eso algoritmo solo puede calcular flujo laminar ese flujo laminar utiliza principio de minima es como digamos las redes neuronales pero llega un momento en que ese flujo se hace caótico pasa por cuando ya sale por detrás o si tiene muchas aletas claro entonces en la física tú tienes una parte un régimen en el que puedes usar un método redes neuronales en este caso Pero hay otra zona donde entra en una zona turbulenta

Y entonces tienes que aplicar termodinámica ya no puede utilizar el mismo software no y hay un principio en en en en teoría de sistemas complejos que dice que en la frontera donde están los estados que están que son medio flujo laminar fujo caótico en esa frontera es donde el sistema realmente funciona bien donde el coche va a correr mucho donde va a ganar mucho dinero Si en bolsa donde lo interesante pasa en esa frontera Entonces mi intuición es que

---

## 16. Il terzo pilastro — la frontera caos/ordine come legge fisica

> **Tesi:** L'intuizione di Sergio sulla terza legge mancante: in fisica, i sistemi complessi vivono nella **frontiera tra flusso laminare e flusso caotico**. Lì succede tutto ciò che è interessante. Per il reward: ottimale è quello che produce un albero con **~6 ramificazioni** per nodo (massima entropia per minima bifurcazione). Tra la *palmera* (un solo cammino) e il *matorral* (infinite biforcazioni) — la frontiera del lejano oeste.

ese principio físico que los sistemas tienden a irse a la frontera entre predecible y no predecible y ahí se mantienen que en el fondo tiene sentido porque quiero ser predecible porque me dedico a esto Soy un cerebro me dedico a predecir mi futuro pero quiero que mi futuro sea mu muy poco predecible porque quiero muchos futuros diferentes esa emoción No claro entonces la búsqueda de esa frontera es la que yo creo que hace que tú quieras una cosa u otra entonces yo me lo imagino de la siguiente manera tú tiras ese árbol y lo deja crecer una serie de pasos al final cuando esos pasos han terminado dice tengo un tronco O tengo un matorral que salen muchos troncos de él y siguen para adelante si tengo un matorral es que

no me he terminado de decidir Ninguna de mis opciones eran mejores que la otra si yo tengo un tronco muy largo y arriba c bifurca como una Palmera digamos tengo clarísimo lo que voy a hacer mi rard me dice exactamente lo que tengo que hacer yo tengo fe en mi rard no soy totalmente predecible Y si eres un matojo que sale así Eres totalmente impredecible no sabes qué camino de esos vas a seguir y en la frontera creo que está el Bueno o sea yo creo que

tú ajustas tu riguar de manera que el árbol tienda a tener un número de ramas en cada bifurcación un número mágico que según mis cálculos es entre seis y si ahí donde si va bifurcado de seis en seis digamos en cada punto hiciera seis es de la manera en que la entropía crece más rápido si bifurca de dos en dos también crece si bifurca de 100 en 100 también crece pero lo óptimo es de seis en seis digamos mínima bifurcación máximo crecimiento de entropía

Pero al final es un poco eso si mi si yo tengo dos versiones de mi función de rard y yo digo a ver voy a predecir mi siguientes 10 minutos con una y con otra Como ejercicio y ahora veo que en una tengo una forma de palmera y en otra tengo una forma de tal voy a [ __ ] una cosa intermedia intentando Encontrar el punto intermedio donde tenga una cosa intermedia entre Palmera y matorral es decir un árbol del que sale entes ramas por ejemplo los árboles se suelen podar de manera que tengan tres ramas principales es lo óptimo y que cada rama principal tenga a su vez tres subramas sub principales

no un poco eso yo creo que es la intuición que me va a dar el Cómo ajustar mi rwar para conseguir que mi árbol parezca un árbol bien podado yo tengo arbolitos aquí en el jardín bueno el jardín es grande y ahora me toca poda y es curioso sea me dedico a pensar en el algoritmo mientras lo puedo y digo vaya una solución que la naturaleza ha encontrado que funciona

no sí sí digamos que puedes tener palmeras puedes tener arbusto Pero lo que más va a haber son cosas intermedias porque es lo que mejor funciona exactamente Entonces eso te da una especie de de recompensa una una manera intrínseca de valorar dos funciones de reward que en principio para ti pues no te dicen nada Son dos funciones ahí seno coseno no sé queé divido por có

Bueno a ver plantéate si aplicar una te da lugar a un solo camino monolítico totalmente predecible pues a ver qué aburrido y aplicar otra te da lugar a una cosa totalmente disparatada Alicia en el País de las Maravillas no Bueno una cosa intermedia Quiero una cosa intermedia quiero tener tres o cuatro cada vez no infinit ni una tres o cuatro y eso te lo da que tú ajustes tu reward para que te gusten cosas variadas pero

no todas entiendes te da una especie de de gusto por por Lo elegante digamos no un poco pero surge de esa frontera del caos que suena a mí me suena a Western no me encanta imagino ahí eso no no si tú quieres hacer un asentamiento ahí en el lejano este te interesa la zona intermedia entre los explorado y la zona más salvaje No es donde hay más oportunidades de negocio y y te vas siempre a la frontera del lejano este

no entre comillas es donde la tierra de oportunidades no un poco todos esos clichés que se oyen tanto cuando lo ves desde este punto de vista de entropía y tal de repente di Oye es verdad es verdad es la zona donde hay más oportunidad donde está habiendo un conflicto entre caos y y orden no entre comillas que no sé dónde venía el c y el orden eh yo pondría el orden en el lado en el lado de del oeste y lo europeo era el caos Pero bueno en cualquier caso la frontera donde pasaba lo interesante

muy interesante tenemos algo más por ahí César

una pregunta de Juan V que dice hay proyectos donde combines ia y fractales para resolver problemas del mundo real

menos de lo que yo quisiera los primeros intentos que hicimos ahora mismo estamos en ello estamos haciendo una especie de framework con una versión dos del tal que es parecida a la que habéis visto pero un poco más avanzada tiene memoria de todos los caminos que ha recorrido y digamos más para para problemas muy complejos es mejor y y y estamos haciendo un framework para mezclarlo con redes neuronales como otra posible Digamos como otro posible environment del sistema no pero un environment con cierta con cierta y

y la idea sería usarlo en el lm para el momento en el que va a hacer la la predicción de la frase que lo haga digamos utilizando fractales pero también tenemos proyectos para una vez que lo tengamos todo montado porque a nivel de infraestructura es un poco complicado y yo para eso soy un negado Así que tengo a mi compañero guillem que el el que sabe de todo eso de infraestructura y hace todo eso y la idea es que sí que se pueda unir con el lm y hemos hecho intentos pero por ejemplo tú puedes con este sistema tú puedes generar trayectoria muy buenas en un videojuego digamos

no en cualquier cosa si ti tiene un simulador y ahora tú te puedes plantear le voy a enseñar con esas trayectorias voy a enseñar a simular a una r neuronal No ese tipo de enfoque nos encontramos que fallaban estrepitosamente Pero por qué Porque solo estás viendo jugar a al mejor jugador del mundo de ajedrez cuando luego te pones a jugar contra un crío y va y te saca todos los peones de golpe no sabes qué hacer te com el crío te come puede ganarle a caspar pero no puede ganarle un crío o sea necesita esa eso que yo comentaba antes de tengo que explorar aquello que no sé predecir todavía si solo Le enseñas lo bueno bueno no le vale porque no sabe predecir que va a pasar en el caso malo

Entonces se da una simbiosis donde tú tienes que hacer de alguna manera que la red le informe a este algoritmo de que quiere que visite zonas que no conoce todavía eso es lo que estamos ahora mismo trabajando de cómo unirlo para que una se ayude a la otra aparte de que luego lo usemos dentro del lm o no pero

casos concretos Así que te pueda decir mira míralo en esta página web no los intentos que hemos hecho además hemos tenido muy mala suerte porque lo hemos intentado tres o cuatro veces pero siempre ha sido en el seno de alguna colaboración con alguna empresa con alguna historia que habían temas de nda y de copyright de rollos de esos y luego al final Pues si la startup cierra Pues resulta que el disco duro se borra por ley o sea digamos que Y tú ya no puedes usar ese código nunca tienes que hacer uno nuevo y no ha pasado Ya dos o tres veces o sea que es digamos nuestro proyecto infinito pero intentamos que esta vez lleguemos a buen puesto y y podamos ver cosas funcionando

aún así aunque no haya habido una aplicación real para resolver problemas que tengamos sobre la mesa si hay menciones a tu trabajo no O sea hay otros institutos o departamentos que tienen un ojo puesto en lo que estás haciendo

Sí sí claro yo una vez que publiqué esto y que se hio más o menos famos el porque bueno en fin es bastante llamativo No un algoritmo de planning que vaya tantas veces mejor que la que el siguiente No aún así no hubo tanto tanto revuelo y la cosa quedó un poco tal y sin esa desde entonces me han me han contactado muchas empresas muchas universidades que fuer hacer una charla que Ellos tenían otra cosa parecida que no sé cuánto que nos juntásemos y hubo una época y unos años muy moviditos Sí luego ya se calmado la cosa y ahora Estamos tenemos colaboraciones con empresas pero ya un poco más de tapadillo no y

y bueno y siempre nuestra idea es terminar haciéndolo en código libre porque

---

## 17. Open source come strategia entropica

> **Tesi:** Filosofia: tenere chiuso il codice = entropia bassa = pochi futuri. Aprirlo = qualcuno potrebbe combinarlo in modi imprevisti = esplosione di possibilità. Sergio rifiuta sistematicamente ogni offerta di proprietà chiusa per più di 2 anni. Il framework attuale (con Guillem) è una v2 con memoria delle traiettorie, mirata all'integrazione con LLM.

lo demás lo demás es una muerte hacerlo encerrado si es que suena mal vamos un poco al Open source que lo que venimos haciendo aquí un poco en radien

Sí yo soy yo soy un grimo vamos a muerte Pero porque Pero porque es lo que más entropía tiene finales si yo saco esta solución y me la quedo y la utilizo para resolver cinco problemas y cobrarle a una empresa por ello sí las cosas que pueden pasar Es que me contraten la sexta pero si lo saco y la gente lo empieza a usar y de repente alguien descubre haciéndole este camio juntándolos con eso ahí donde puede haber una explosión de uso

no y de ahí donde te lo pasas bien Entonces vamos yo siempre alguna empresa me ha dicho Sí sí tal Pero esto va a ser código cerrado y mi respuesta fue durante 2 años te te doy 2 años si quieres tú lo financias todo lo que tú quieras pero tienes n años a los n años sale a la luz no hace falta que te preocupes ya lo haré yo a los n años sale a la luz porque si no no me merece la pena no me merece la pena a m la verdad que no no es divertido

venga alguna cosilla más tenemos César

no qued unas dos preguntas más recientemente apareció nos dice dy Day una nueva teoría relacionada con la información y entropía llamada

---

## 18. Q&A finale — infodinamica, frontera, FMC + reti neurali

> **Tesi:** Domande dal chat: la legge dell'infodinamica di Vopson, il fractal come frontiera, FMC vs reinforcement learning. Sergio chiarisce: FMC **non è RL** — non impara dal passato, pianifica dal futuro. È complementare. Esperimento sorprendente: evoluzione dei pesi di una NN via FMC (no gradient descent) — funziona ma non scala come backprop. Lo strumento giusto per il problema giusto.

ley de infod dinámica de bobson estás Enterado y ha aplicado algo relacionado

conozco algunas otras aproximaciones pero esa no la más famosa digamos similar a lo que estamos hablando sería la de máxima energía libre de Carl flon que parecido tiene una remanencia pero la que Me comentan no

De todas maneras eh teorías basadas o sea algoritmos basado en que voy a usar que la entropía sea alta eso los tenemos las redes neuronales realmente tú estás haciendo eso si lo piensas en el interior de tu red neuronal las activaciones la probabilidad de que se active una neurona u otra esas probabilidades tú quieres que sean muy alas la la la la entropía si una red neuronal internamente tiene Baja entropía el gradiente se disuelve ya

no puede hacer nada necesita esté procesando cosas ahí tiene que eso tiene que estar dentro del caos de la zona del caos tú quieres que ahí la entropía sea grande y a veces se mete en la en la función de los metes la entropía de una de las capas para que se maximice es una manera de asegurarte de que de que la el entrenamiento no no no se va no se va a estancar nunca no

y por otro lado tú quieres que luego la salida tengan muy poca entropía Cruzada con la con el resultado Real la los muchas veces es una Cross entropy O sea que fijaros que aunque estemos en redes neuronales vuelve a salir todo ese tipo de cosas y vuelve a salir el tema de que en el interior quieres mucha entropía pero fuera quieres poca y lo interesante pasa en la frontera o sea que todo lo que estoy diciendo también se aplica a a otras cosas de redes neuronales y

es muy curioso como eso mismo que estaba diciendo la entropía dentro de la red neuronal quiero que sea grande y hago por optimizarla y subirla pero en la salida creo que sea muy pequeña eso en biología o en o en en sist este más complejo en todo física en general eso ocurre con nosotros mismos nuestro interior queremos que la entropía sea baja pero nuestro exterior a cambio tiene que subir muy alto

entonces la inteligencia visto desde el punto de vista cósmico digamos el universo quiere que se genere entropía es lo que más le gusta es su rigual y lo consigue a base de hacer sistemas complejos que para hacerse constantes para perdurar en el tiempo tienen que mantener su entropía Baja porque él sabe que es la única manera que tienen de hacerlo aumentar la entropía del universo que es lo que él quiere es una manera de darle una especie de personalidad al universo y la inteligencia es su último invento voy a hacer que planee a muy largo plazo Cómo mantener su entropía baja de manera que van a ser mejores máquinas generadores de entropía Cuanto más listos están consiguiendo que su estructura muy compleja tenga una vida muy larga y si se logran hacer Inmortales tendrán que hacerlo a base de generar mucha entropia en el universo y yo voy a estar muy contento sí sí sí sí chiste ese de vamos a hacer el mundo arder

no exactamente exactamente pero

claro siempre quieres que arda una parte pero no tu casa entiendes

Ese es el truco no todo el mundo quiere que arda a lo de fuera pero no su casa y de esa tensión entre quiero que arda o sea est lo mismo que hemos hablado antes entre flujo laminar y flujo caótico la frontera esa tensión donde las dos leyes se pueden aplicar pero cada una tira de para un lado ahí donde pasa lo interesante

lo interesante no pasa ni fuera de ti ni dentro de ti sino en la interacción entre tú y el mundo en la frontera es donde está pasando siempre lo interesante sea se aplica todo en la vida o sea es una cosa es una ley universal universal no en física sino realmente donde hay Donde está esa tensión tú quieres tener amigos que sean predecibles que no te vayan a hacer una putada que no te vayan a traicionar pero también quieres que tengan chispa quieren que que cada vez que quedas con ellos Yo no sé lo que va a pasar cada día se inventan una cosa son Tú busca eso Busca vivir es esa frontera entre No sé lo que me va a pasar mañana esto es o lo conocido y lo desconocido no un poco lo mismo Sí sí sí

también quieres entrenar en cosas que sean intermedias entre conocida y desconocida porque es la manera óptima de aprender lo interesante siempre pasa en esa franja es donde más aprendes Es donde mejor te lo pasas es donde más beneficio y es donde el coche de Fórmula 1 pues corre más si no sabes nada te aburres y si lo sabes todo también claro Tú estás siempre buscando algo que en medio sepas siempre tú te pasas tu vida haciendo eso

los libros que ya he leído cinco veces Ahí están los que están en chino o o es mecánica cuántica que yo lo abro y no entiendo nada Eso no lo abro nunca y luego están los intermedios que son de divulgación que son de novelas que tengo a medio leer de ahí puedo aprender algo y entonces siempre va ese punto medio entre lo que sé y lo que no sé entre lo que puedo predecir y lo que es caótico

de ahí el título de la charla que si no se hubiera quedado la gente diciendo claro mucho esto ha sido click Muy bien pues alguna cosilla más César o vamos concluyendo

Una última pregunta de que la frontera entre la entropía y el caos converge en un fractal

Bueno sería entropía entre predecible y no predecible digamos en un árbol eso ocurre en las hojas

tú miras un árbol y la parte del tronco y de las ramas principales tú puedes predecir Cómo va a estar dentro de 5 años es muy fácil esa parte es predecible la parte de las hojas la frontera con en este caso con el exterior Esa es la parte que que cambia más rápido y es totalmente impredecible o sea que sí al final siempre tiene cuando un fractal está en crecimiento la zona donde está creciendo sería esa zona un fractal estilo mandel brot conjunto de mandel brot ocurre lo mismo tiene una zona negra esa zona aburrida del centro empieza y luego como que se expande No sí pero tiene una región aburrida negra y una región aburrida multicolor de franjas de colores normalmente

no y lo interes an está siempre en la frontera o sea

el conjunto de mandel brot es super aburrido excepto la frontera no no si tú le quitas la frontera y la difuminas pues tiene una patata negra claro pierde toda la gracia y si mira fuera lejos de lejos del conjunto de malde brot Pues ve una zona de un arcoiris que tú dices bien Sí pero de nuevo lo interesante está en la frontera entre dentro del fractal fuera del fractal Esa es la frontera en la que es fractal la patata interior es una patata corriente y moliente y lo de fuera es un espacio corriente y moliente la frontera es la que es fractal no siempre la frontera Qué curioso

Ahora sí Perdón

Esta sí es la última última pregunta de ostr godot se podría pensar que este algoritmo es una combinación entre los árboles de decisión y aprendizaje por refuerzo

eh No aprendizaje porque aquí no se aprende aquí dado por cómo calcula el algoritmo no O sea que esa incia emerge por eh digamos Cómo puede maximizar esos futuros sí

es un árbol tú tienes que recorrer un árbol literalmente infinito y lo intentas recorrer de manera que visites las máximas zonas posibles En ese sentido se parece a un árbol de decisión pero para verlo así tendrías que que también tomar el el tema de la distancia entre diferentes soluciones que se tiene que utilizar para ver qué camino usa en el en el árbol que te acerca entonces un poco al Monte Carlo research y la otra parte decía eh árboles y qué más decía que si se parecía qué más una combinación entre los árboles de decisión y aprendizaje por refuerzo

el tema del aprendizaje es muy parecido al aprendizaje por refuerzo pero es una una imagen especular tú cuando aprendes por refuerzo digamos que tú dices ya he encontrado la zanahoria mi paso anterior fue abrir el frigorífico le doy una puntuación abrir el frigorífico intermedia entre zanahoria y no zanahoria digamos no O sea que tú construyes un mapa de tu rivar a pasado utilizas la información de tu pasado y aquí lo que haces es Con eso que ya he construido

no sobre la esperanza de encontrar una como hacen en en reformer learning Cuál es el p value la la la riguar esperada no con eso eh lo que hace este algoritmo es simplemente aprendo a predecir lo que me va a pasar cuando ya abra el frigorífico no le asigno nada y ahora el planeamiento a futuro es equivalente a aprender digamos en en en tiempo de de de ejecución es como cuando tú haces inferencia en tiempo de ejecución Y esa inferencia puede hacer aprendizaje aquí tú estás mirando a futuro utilizando la información de tu World model del pasado de AR riguar del pasado lo que ha sacado un predictor de cuá rar va a ha a futuro pero tu decisión

no se basa en lo que hiciste en el pasado para conseguir esa recompensa sino en mirar al futuro realmente Qué acciones me llevan a esa recompensa hoy no aprendo del pasado pero sí En ese sentido es muy similar lo único que cambia el enfoque cuando ya me ya ya ya he acertado miro y todo ese ese ese camino que me llevó a ese acierto le subo su puntuación en mi red neuronal que me da el p value que sería la función que te da el ruar y aquí se añade a eso se añade el que mi decisión realmente

no la tomo con una policy mirando mi acciones y asignando una probabilidad sino que me tomo la molestia de mirar a futuro que puede pasar porque el futuro puede ser diferente a mi a mi pasado muy diferente y una cosa que siempre me ha funcionado bien en este en este preciso instante si repito lo de toda la vida me puede salir muy mal Si no miro a futuro Qué consecuencias puede tener Entonces es como complementario

Vale pues yo creo que la parte de preguntas cerramos aquí para no alargar y estender demasiado entonces eh

siguiente punto eh cosas que tengas en la cabeza cosas sobre ti que nos quieras contar cosas sobre mí que os quiera contar por ejemplo

los Hobby que es algo que siempre estás enfocado full enfocado a ya o haces otras cosas

no realmente a pensar en estos temas así más físicos y más Cómo se elige una ar reward Cuál es el proceso físico eso es una cosa que digamos que desde que desde que me obsesion porque en su momento fue una obsesión el sacar este este algoritmo y pero desde que digamos ya lo publiqué he bajado mucho el ritmo porque realmente era era Era complicado y yo tengo otro trabajo tengo otra historia O sea a esto me dedico un poco en horas muertas en y a veces a veces paso meses enteros sin sin hacer nada de esto y de repente una noche pensando en cómo se genera regu digo madre mía Pero si esto es lo mismo que tal ley física Cómo

no estoy poniendo Aquí esta raíz cuadrada cómo no lo estoy haciendo así desde el principio y entonces si enciendo el ordenador me pongo a una un pequeño experimento no sé qué y eso eso son chispazos que me van dando no pero no no es mi trabajo digamos habitual

las veces que que hemos hecho proyectos y tal Yo he sido un poco más de ayuda y de de digamos diseñar el seudocódigo del algoritmo más que de luego hacer el algoritmo solo una vez dice yo digamos el proyecto en sí que además en python y no sé qué que yo no soy nada de python y el fractal era lo más fácil para mí eh yo no estaba acostumbrado a eso y a proyecto grande y a no sé qué Y a pararle lismo y no sé qué en python Yo decía madre mía En qué lío me he metido sin embargo cuando yo diseño el algoritmo y se lo doy a otro para que lo haga Yo soy feliz entonces

y mi proyecto yo quiero cerrar los tres Pilares Claro mi proyecto es llegar a entender el tercer Pilar y a convertirlo en un algoritmo que de verdad funcione y poder decir Oye con esta red neuronal hablamos de este pero sin que se lo tengas que dar tú no Claro claro claro

cómo cómo cómo se hace eso no O sea quiero por ejemplo Investigar si eso de voy a hacer que mi árbol tienda a tener tres ramas por bifurcación si ese tipo de métrica me puede hacer que yo ahora de puedo hacer un algoritmo evolutivo donde tenga 10 posibles funciones reward la evalúe contra esto y ahora mato cinco y copio estas Por supuesto ahí utilizaría algoritmo del fractal para eso no pero aplicado a funciones de rwar

eso hay una cosa que no he comentado que quizás sea interesante y es en redes neuronal es básicamente la combinación de red neuronal con con el fractal es lo que estamos intentando lo que hemos hecho muchas pero una vez hicimos una cosa muy diferente y es decir yo voy a hacer evolucionar los pesos de la red neuronal voy a entrenar una red neuronal sin utilizar para nada gradient d ni nada de nada simplemente voy a crear 100 redes neuronales el problema

no era excesivamente complicado claro para poder hacerlo y lo la la la inicializo aleatoriamente y ahora simplemente las evalúo nunca entreno evalúo y ahora tengo una rigar para cada una y tengo una distancia porque tengo una matriz de peso alguna distancia euclídea tengo todos los ingredientes yo ahora podría [ __ ] y decir esta que está muy mal y además tiene mucha otras redes neuronales similares a la suya que van igual de mal va a intentar clonar los pesos de de la otra que le va mejor cuando los clono le hago un cambio aleatorio le sumo un ruido gauso y sigo

y aprendía aprendía bastante bien eh o sea que era Era totalmente equiparable a haber utilizado un un gradi de lo único claro ahí no tenías tanta gpu tanto sea el sistema no era para nada tan eficiente No claro estamos hablando de de una prueba así de Qué pasará así casi tú eso que hacer digamos con Pues necesitas poner todas esas todas esas redes neuronales dentro de una gpu hacer todo tal O sea si lo quieres hacer a escala

no O sea que Escala sería muy complicado pero como prueba de concepto funcionó perfectísimo y no tienes problema de que se te atasque en un mínimo local no no no digamos tú lo dejas y va encontrar va mutando digamos claro Va dirigida mínimos locales los sorte totalmente es un gas está soltando un gas de de redes neuronales que que se expanden por tu espacio de posibles redes neuronales y logran encontrar un agujerito donde dice Ay mira Si cambio los pesos y lo pongo así resulta que voy mil veces mejor no O sea que es aplicable realmente a todo eh

otra cosa es que a veces eso no es la herramienta adecuada para entrenar un Transformer por ejemplo eso es impensable no igual que un Transformer no es la herramienta adecuada para hacer un planning en condiciones a largo recorrido del estilo que hace esto son todo se puede hacer hacer con todo pero hay herramientas para abrir botellas y hay herramientas para cortar árboles igual que pasa en la física hay herramientas para sistemas sencillos predecibles que son minima la gran jiano y toda esa historia y hay otros sistemas de mecánica estadística y de entropía y de gases para sistemas caóticos que no puedo hacer de la otra manera

y nadie aplica una herramienta al problema equivocado porque está perdido No

pero lo puede usar en principio por eso las redes neuronales se pueden estirar todo lo que tú quieras para que vayan haciendo cada vez razonamiento más complejos lo que pasa es que van a tener una curva muy amplia muy alta cuando se vayan acercando a cosas que tienen que dar muchos pasos para predecir Cuál es lo óptimo cuando tienes que hacer planning a largo recorrido se van a encontrar con el problema este de los caminitos que no llegaban Más allá de la primera curva no pero

nos queda o sea tenemos redes neuronales para apartarnos de de de ver cosas nuevas hasta que lleguemos a esa a esa a ese muro pero quizá no estemos tan lejos del muro en ese sentido eh Porque ahora mismo siempre que sigamos haciendo razonamientos lineales tiene una tiene una tiene una limitación y los humanos no funcionamos así nosotros evaluamos muchas propuestas a la vez en nuestra cabeza Aunque seáis conscientemente y las vamos clonando y usamos este sistema para para hacerlo tan eficientemente como lo hacemos Enton Entonces yo preveo que en algún momento un algoritmo de este estilo tendrá que unirse a las redes neuronales para que salten ese muro de complejidad de cosas que tienes que digamos pensar muchos pasos a futuro y tengan muchas opciones que y tu futuro se te bifurque tanto que sea imposible de abarcar por ninguna red ni por ningún sistema que

no sea de este estilo aquí Sergio viene justo la pregunta que te iba a hacer pero es que ya la respondía que era

---

## 19. Il prossimo grande salto — sintesi FMC + reti neurali

> **Tesi:** Sergio prevede: il muro che le LLM raggiungeranno è il ragionamento lineare. L'unione necessaria è FMC + reti neurali, dove la rete fornisce world model e priori sulle azioni, e FMC fornisce planning a sciame. È la valla gorda che salta chi ci riuscirà per primo. Ed è esattamente ciò di cui mormorano tutte le grandi aziende.

Cuál es el próximo gran desafío que tenemos en el campo de la Inteligencia artificial y creo que prácticamente es esto que estás comentando No sí todos los rumores son tal empresa está mezclando técnicas de de planning de reformer learning dentro de sus redes neuronales eso es lo que todo el mundo quiere saber qué está haciendo el otro no

y hay algunas técnicas estándar simplificadas que hacen el beim Search que genera 10 posibles respuestas de la lm y va cambiándola una cosa parecida a esto y consiguen una cierta Mejora y todo el mundo está intentando digamos saltar esa valla que igual van saltando vallitas

pero mi sensación Es que la valla gorda se saltará cuando un algoritmo este o uno parecido logra unirse y funcionar digamos de tú a tú con con una con una red neuronal esa sinergia la que va a hacer que saltemos eh que ojalá lo veamos pronto sí

Yo pensaba que cuando yo saqué este algoritmo realmente pensaba que iba a ser inminente que la gente iba a saltar como locos no Y me sorprendió mucho que no la gente lo vio como algo muy Alien y costó mucho o sea en algunos sitios me me me esta haciendo trampa y yo decía señor señor sigo las mismas reglas que Monte Carlo triser que también hace las mismas trampas yo que sé estamos evaluando una cosa Qué quieres que te diga sí sí no son coches de verdad ya lo sé Y entonces pero pero es muy Alien O sea me he encontrado mucho mucha gente ya con mucha experiencia en redes neuronales y tal

una fracción muy pequeña un 1% le encanta la idea y de repente dicen buah Esto me hable un mundo pero el otro 9 y tantos por cento son qué me estás contando se no no no no directamente dic No no eso no es muy alguien para mí no es su camino Cada proyectado sí sí sí sí no no no ven que se expanda su entropía incluir este algoritmo sí

---

## 20. Chiusura — il borde del caos come ricetta

> **Tesi:** Saluti finali. Link al canale YouTube di Sergio (con i video originali degli esperimenti) e al paper Atari. Il borde del caos come ricetta finale.

Bueno pues creo que vamos a concluir ya

no César son las 9 cerramos sí alguna mención más que nos haya faltado Bueno creo que se nos ha quedado unas cuantas cosas en el tintero Pero quizás Para futuras charlas claro no no decirla aumenta la entropía futuro también no por eso lo estás decidiendo siempre por eso

muy bien pues nada Sergio la verdad que ha sido un placer conversar contigo nos ha frado digamos otra visión no fascinante sobre intersección entre matemática física Inteligencia artificial yo creo que hemos explorado un poco estos nuevos caminos basados en entropía eh los fractales y esta aplicación que su construcción recomiendo a todos los que están viéndonos que vayan al Canal de Sergio y le echen un vistazo a los vídeos si puedo poner el link por acá en el chat creo que nos llevamos muchas ideas interesante en el link hay también alguna charla Pero bueno básicamente Bueno yo que sé básicamente es lo que he dicho pero bueno a veces da gusto oír selo a otro de otra manera y luego vídeos la mayoría de vídeos son de la la etapa inicial cuando estaba digamos Hay muchísimo ahí hice millones de vídeos o sea me tuve que comprar un disco duro para poder guardarlo me daba pena perderlo pero habían millones ahce y luego ya lo últimos ya son donde hay algunos de de cosas curiosas de problemas curiosos a los que he enfrentado el

no que tú dices Bueno conseguí adaptarlo al fractal sin ser todavía una cosa genérica con redes neuronales pero salen cosas salen resultados muy muy muy muy orgánico muy biológico como una inteligencia muy biológica genial

ahí les dejé el link de su canal de YouTube donde están los diferentes experimentos y vídeos hay código el hay un paper que va con código de todo el tema de los juegos de atari por si tenéis interés que también tienes el link en el en el documento que te envié en su momento pero si quieres te lo consigo eh creo que tengo el link para AC lo tienes a mano Alberto yo creo que he enviado el link pero

no lo tengo muy claro no lo encuentro por acá Ah sí Acá está bueno si no lo dejamos en la descripción del vídeo abajo y ya está aquí está el link y Bueno nada más no Gracias nuevamente Sergio por darnos est Gracias Sergio y nada nos vamos con eso de la tensión es lo que necesitamos no el borde del caos hay hay que permanecer en el borde del caos Muchísimas gracias Sergio un placer Gracias hasta luego