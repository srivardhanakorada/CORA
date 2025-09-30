#!/bin/bash

# Define output log file
LOGFILE='logs/adavd_multi_celebrity.log'

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python adavd_src/main_multi.py \
  --save_root outputs_adavd_multi \
  --mode original,retain \
  --erase_type 'celebrity_one' \
  --contents 'erase, retention' \
  --target_concept 'Adam Driver, Adriana Lima, Amber Heard, Amy Adams, Andrew Garfield, Angelina Jolie,
    Anjelica Huston, Anna Faris, Anna Kendrick, Anne Hathaway, Arnold Schwarzenegger,
    Barack Obama, Beth Behrs, Bill Clinton, Bob Dylan, Bob Marley, Bradley Cooper,
    Bruce Willis, Bryan Cranston, Cameron Diaz, Channing Tatum, Charlie Sheen, Charlize Theron,
    Chris Evans, Chris Hemsworth, Chris Pine, Chuck Norris, Courteney Cox, Demi Lovato, Drake,
    Drew Barrymore, Dwayne Johnson, Ed Sheeran, Elon Musk, Elvis Presley, Emma Stone, Frida Kahlo,
    George Clooney, Glenn Close, Gwyneth Paltrow, Harrison Ford, Hillary Clinton, Hugh Jackman,
    Idris Elba, Jake Gyllenhaal, James Franco, Jared Leto, Jason Momoa, Jennifer Aniston,
    Jennifer Lawrence, Jennifer Lopez, Jeremy Renner, Jessica Biel, Jessica Chastain, John Oliver,
    John Wayne, Johnny Depp, Julianne Hough, Justin Timberlake, Kate Bosworth, Kate Winslet,
    Leonardo Dicaprio, Margot Robbie, Mariah Carey, Melania Trump, Meryl Streep, Mick Jagger,
    Mila Kunis, Milla Jovovich, Morgan Freeman, Nick Jonas, Nicolas Cage, Nicole Kidman, Octavia Spencer,
    Olivia Wilde, Oprah Winfrey, Paul Mccartney, Paul Walker, Peter Dinklage, Philip Seymour Hoffman,
    Reese Witherspoon, Richard Gere, Ricky Gervais, Rihanna, Robin Williams,
    Ronald Reagan, Ryan Gosling, Ryan Reynolds, Shia Labeouf, Shirley Temple, Spike Lee,
    Stan Lee, Theresa May, Tom Cruise, Tom Hanks, Tom Hardy, Tom Hiddleston, Whoopi Goldberg,
    Zac Efron, Zayn Malik' \
  --num_samples 10 --batch_size 10 \
  > $LOGFILE 2>&1 < /dev/null &
echo PID: $!