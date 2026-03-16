* [ ] For each block, the last step should not move any experts
* [X] Add benchmark code
* [X] Strategy to try: move expert to GPU first and then calculate tokens instead of predict experts that will be used. Try to let GPU compute tokens as much as possible
* [X] Strategy to try: predict and move experts every other steps
* [ ] Eval llada2-flash
* [ ] Block wise: use previous block as next predicted experts
* [ ] Use different CPU expert/GPU expert ratio
* [ ] Notice that the second half
