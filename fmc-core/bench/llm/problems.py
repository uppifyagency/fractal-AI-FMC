"""Small custom math benchmarks for Bet 2 (Fractal-of-Thought).

Two tiers:
  EASY     — single-step elementary reasoning. Most 1B models solve at 100%.
  HARD     — multi-step / 2-3 reasoning hops. Where greedy can fail and
             diversity-based methods may help.

Self-contained to avoid HuggingFace dataset downloads.
"""

EASY = [
    ("If a train travels 60 miles in 90 minutes, what is its speed in miles per hour?", 40),
    ("There are 12 students in a class. 1/4 of them are absent. How many are present?", 9),
    ("A book costs $15. After a 20% discount, what is its price in dollars?", 12),
    ("If x + 5 = 17, what is x?", 12),
    ("A rectangle has length 8 and width 6. What is its area?", 48),
    ("If 3 oranges cost $2.40, how much do 5 oranges cost in dollars?", 4),
    ("A car uses 5 liters of gas per 100 km. How many liters for 250 km?", 12.5),
    ("What is 15% of 200?", 30),
    ("If I save $50 each week, how much money will I have after 8 weeks in dollars?", 400),
    ("A pizza has 8 slices. After eating 3 slices, how many slices remain?", 5),
    ("Sarah has twice as many apples as Tom. Tom has 7 apples. How many apples does Sarah have?", 14),
    ("A triangle has angles of 60 and 70 degrees. What is the third angle in degrees?", 50),
]

HARD = [
    # Multi-step reasoning chains where small models often go wrong.
    ("A bookshelf has 5 shelves. Each shelf can hold 8 books. If we currently have 28 books, how many more can we add to fill all shelves?", 12),
    ("Tom is twice as old as Sarah. In 5 years, Tom will be 25. How old is Sarah now?", 10),
    ("A water tank holds 240 liters. It leaks 4 liters per hour. After how many hours will it be half full, starting from full?", 30),
    ("Anna bought 6 pencils at $0.40 each and 3 erasers at $0.75 each. She paid with a $10 bill. How much change did she get?", 5.35),
    ("A train leaves at 9:15 AM and arrives at 11:50 AM. The trip is 200 km. What is its average speed in km/h?", 77.42),
    ("In a class of 30 students, 18 study French and 14 study Spanish. 7 study both languages. How many study neither?", 5),
    ("A jacket originally costs $80. It is on sale for 25% off. After the sale, an additional 10% tax is added. What is the final price?", 66),
    ("A father is 4 times as old as his son. In 6 years, the father will be 3 times as old. How old is the son now?", 12),
    ("A box contains red and blue marbles in a 3:5 ratio. There are 32 marbles total. How many are blue?", 20),
    ("A tank fills at 5 liters/min and drains at 2 liters/min. Starting empty, when will it have 60 liters?", 20),
    ("A rectangle's perimeter is 36 cm. Its length is 4 cm more than its width. What is the width?", 7),
    ("A car traveled 180 km using 12 liters of gas. At this rate, how much gas is needed for 750 km?", 50),
]

# Default benchmark used by main(): EASY + HARD = 24 problems.
PROBLEMS = EASY + HARD
