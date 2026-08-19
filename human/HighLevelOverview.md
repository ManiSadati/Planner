# Projects overview

there are four main repos:
https://gitcode.com/Ascend/AscendNPU-IR
https://github.com/hw-native-sys/PTOAS
https://gitcode.com/cann/pto-isa
this repo: https://github.com/ManiSadati/Planner

so there are two repos, NPUIR and PTOAS. both of them try to lower from high level frameworks like triton or pypto/tilelang, to something like cce which is very low level.

for NPUIR, they start from a triton code, they do fusion planning and fusion, then synchronization and memory planning and while doing those things they also lower their IR gradually, until they get to HIVMAVE intrinsics. from there, it will be converted to llvm which would basically mean its the input of ccec and its cce code.

for PTOAS, they rely on pypto a lot as their main input framework, they have fusion planning synchronization and memory planning, then they lower it to vpto i think and at that time they will do the low level fusion and bunch of stuff very low level. they are trying to be basciaaly the same as ccec that does those very lowlevel stuff. one important thing is ptoas relies on ptoas a lot. for a lot of their examples they just use the synchs and plannings of tiles from pypto and those sync and memplan passes will become a bit optional. they also had two backends emit-C and vpto. emit-C would basically mean they would turn their IR to PTO-ISA and PTO-ISA expands to cce instructions. but i think that was a temporary transition plan. recently PTOAS introduced a new IR for their VPTO backend. so before that you would deal with tileops up to a certain point and then it would expand to vpto (ptodsl tilelib). but now i think either from the start it would be vmi or after some lowering youll get vmi and then vpto. note that vmi is only for vector instructions.

also i forgot to mention there is a PTO-ISA repo as well. ptoisa is like a virtual isa above cce that works with tiles. so same idea as ptodsl but it lowers into cce intrinsics.

This repo (Planner) is basically for planning and documenting our progress and would be as the main context for all these repos. there is a explorer agent in this repo that will give us daily updates on the issue pages, PRs and forks of PTOAS and possibly npuir so that we know if there is going to be a conflic in terms of design or a major change.

making sure ptoas is not doing a major update is very important. the Planner/explorer job is to verify we are aligned with both npuir and ptoas in terms of big design. explorer should work like an exploring agent bringing update from ptoas or maybe npuir and see whether these updates align with us or not.


# Plan overview
So the main plan is that obviously both NPUIR and PTOAS are doing the same thing. so whats a good way to unify them. we basically aim to replace the ccec part of npuir with ptoas. that means npuir will generate and lower up to a point and then we convert the rest to ptoas. we are looking at ptoas as a open source cce so you know thats a main point of doing this to have everything open sourced.

so the overall solution is to create lowering path for npu-ir through pto-isa for backend lowering (HIVMAVE to VMI IR).

PTO (VMI): virutal ISA, not tightly coupled to HW. scalalbe for future hardware architecture, VMI handles the vector side, for Cube side and DMAs it would be a bit more complex, either we will transform them to pto dialect instructions (like pto.mte_gm_ub) or rewrite the CCE templates in NPUIR.

for the conversion we are trying to modify NPUIR side and have our pass over there, not PTOAS.
we also have to make sure that npuir wont lose performance by lowering from ptoas.

## current stage
so we looked at the NPUIR dumps and it seems like the convert-hivmave-to-ave-intrin is the pass that should be replacement start from. basically this pass and all the passes after it would be replaced by flow of PTOAS. we need to take a look at PTOAS again but seems like for the PTOAS side everyhing before the expandtile op pass could also be ignored and pipeline of conversion starts from after that pass. we need to double check especially with the vmi. its been a while i didnt get to take a look at it.

so for the conversion we are trying to look at HIVM-AVE IR and try to make a one to one mapping of that to PTO VMI. so memrefs would mainly become pto.ptr or hir.vload would become vmi.vload. hir.vadd would be pto.vmi.vadd and ave.hir.pge would be pto.vmi.create_mask roughly.

Update: Right now we already added a lot of stuff to NPUIR, including the pto dialect library, two passes (one for template/DMA conversion and the other one for converting to vmi):
https://gitcode.com/wilsoncxfeng/AscendNPU-IR/tree/melika/ave-to-vmi (I think this is the most up to day one)
https://gitcode.com/wilsoncxfeng/AscendNPU-IR/tree/pto-dialect
https://gitcode.com/wilsoncxfeng/AscendNPU-IR/tree/mani/DMA
There might be newer branches with more updated scripts. all branches will eventually be merged into 
https://gitcode.com/wilsoncxfeng/AscendNPU-IR/tree/master

one important thing right now is to make the master branch of wilsoncxfeng fork to be updated with all the new changes we added to NPUIR.


But of course there are some challenges (I'm not using the full IR instruction here):
1. Synchronization
    - hivm.hir.set_flag would become pto.set_flag (I think somewhat done already)
    - we wont use sync pass of ptoas we use whatever npuir has
2. Memory plan  (I think somewhat done already)
    - hivm.hir.pointer_cast() : memerf will become pto.castptr : -> !pto.ptr 
3. predicate and masking: 
    - predicates represenet as ave.hir.pge <ALL> : vector<128xi1> in the source should be converted to vmi mask values like !pto.vmi.mask<128xpred>
    - PTOAS subesquently determines their physical mask granularity and layout from the govened vector ops.
4. C/V data Transfer & synch
    - cross core syn maps to pto.sync.set on the producers completion pipeline and pto.sync.wain on the consumer's execution pipeline
    - hivm.hir.sync_block_set[<CUBE> <PIPE_M> <PIPE_V>] flag =0 will become sth like pto.sync.set <PIPE_FIX>, 0.
5. DMA Transfer  (I think somewhat done already)
    - NPUR IRDMA trnasfers are representad as memerf and datamovements like hivm.hir.nd2nz and hivm.hir.load/store. in ptoas, DMA is expressed either with tilesops (tload/tstore). target specific tempaltes expand these into PTO MTE ops like mte_gem_l1_nd2nz/mte_gm_ub.
6. Cube insturcions
    - NPUIR side use insturctions like mmadL1 or mma* that use cce templates for lowering. meaning there are some templates in NPUIR that get expanded to cce intrinsics and functions. they wont be lowered like many other ops.
    - for conversion to PTO, i think the best way is to reimplement the on to one mapping of those templates to be lowered into PTO instead of CCE so its a risk/challenge.


one important this for 5 and 6 ( or maybe some of the other points) is that we might need to do the conversion in higher level if its easier to map at that level or make it one to one.

lets treat this fork as the main development for NPUIR to ptoas conversion. https://gitcode.com/wilsoncxfeng/AscendNPU-IR

## Compile and Run flow
So we have two servers, 
1. a server with only simulators that codex/ai does have access to. We recently found out you can run both NPUIR and PTOAS simulators in this server(pretty sure there are some documents in this Planner repo that talk about it, or should be).  
2. a server with A5 machine that codex/ai doesnt have access to. So for development over there we mostly write in the other server push it to github and then use that over there. but transferring IRs or compiler errors from one server to another is a bit hard. Both NPUIR and PTOAS can be run on the a5 server (pretty sure there are some documents in this Planner repo that talk about it, or should be).  


we should have the scripts or docs that talk about them (both how to run and compile) in the NPUIR and PTOAS folders not the bridge folder.
Also its important to have a similar way of comparison.

The things we need for comparison are:
1. environment: they should be run on the same simulator as well. i think its Ascend950PR_9589. you should have a script that sets up the environment for each of the runs for full NPUIR and the other one for NPUIR/PTOAS conversion path. you should have a one line msopprof / msprof op command as well for each of them so that i can also run it myself.
2. num ticks: we need to have the num ticks. for ptoas, its easy msopprof gives you that number but for npuir, its a bit hard we need to figure that out. we need to make sure both ptoas and npuir are using the same number of ai cores as well. Also  we need to make sure we are looking at similar numbers. not comparing apples to oranges.
3. runtime: on A5 machine we should mainly look at the runtime.
4. the triton kernel has already the host code inside it as well. but for the PTOAS we need to write a host code for each test case we need to make sure that host code is compatible and equivalent to what triton/npuir is running.



## future
so the main plan for now is to do conversion from npuir to potas. but in future we may want to explore some deep fusion ideas of ops. for example similar to the idea of online softmax but actually break down the reduction instructiosn and add repair terms either through compiler or specific templates. we also may need to reorder some ops to make it easier to fuse but thats for future.


# Plan breakdown
we mainly need to put our pass in npuir before the convert-hivmave-to-ave-intrin . we may need to have several passes.
HIVMToStandrd, this is the pass wher ethe hir.load / store and mmadL1 instructions gets converted to the cce func template calls. meaning this might be where we need to intercept before ave specifcally to these instructison.

## explorer
    making sure ptoas is not doing a major update is very important. the explorer job is to verify we are aligned with both npuir and ptoas in terms of big design. explorer should work like a cron job that every morning at 7 am looks at all new or newly modified issue pages, PRs and  branches of every fork especially https://github.com/zhendong404/PTOAS, https://github.com/mouliangyu/PTOAS  and https://github.com/WenboCodes/PTOAS and of course every fork of https://github.com/mouliangyu/PTOAS/forks becuase this guy is very active.

# Rules
read ~/Planner/AGENT.md every time!



# Tasks


## Initial stage (Already Done)
this is going to be a very heavy work.

right now codex doesnt have that much background . there are already some good documents in ptoas and npuir, that can be used for getting context but of course those are in different places in both repos. i guess the first step is to bring the importnant docs especially the design docs to Planner. so for example have a PTOAS folder and NPUIR folder and in each of them have a design doc and a coding guide doc folder. for coding guide maybe you dont have to have everything just either map the repo and docs and give a high level overview of pipeline and abstractions.
for the design doc folders try to aggregate the important docs inside the npuir and ptoas and have a summarized but ofcourse detailed  design docs there. note that this should also contain their future plans or anyhting the other forks/branches have not only main branch. ofcourse most of the branches are useless so you dont really have to take a look at them.
right now since we have to setup explorer firsrt, its worth to look at all those forks and branches and PRs and issue pages i mention but not for just today for the whole past month. make sure you look at those forks i mentioned to you.

Make another folder in Planner called bridge. in there you will have memory and planning folder. memory gives you the overal progress, the details of where to code , some good/bad patterns of how to code how to install stuff and run. these kinda stuff but very brief and to the point. also the updates of the ptoas can be connected to here. this folder is the connection between human coder and ai coder.
there is also another folder called human. that one is written by human coder. it sets the main contract of what AI can do and cannot do. AI has to be aligned with it. AI agent should make sure documents in bridge folder align well with documents here. AI should not modify docs in human. if anything doesnt align, you should notify human but be to the point and brief.


note that this intial stage is done by codex not the api key of openai.


## make the AGENT.md (Already Done)
that file should be written soon. it should also look at the explorer folder to see if there is a new update, meaning when im working with codex, before you actually answer the prompt if its the first prompt of the day , you just summarize what happened that day in the explorer side and if i asked more or you think it was a big thing add it to the overview readme of explorer. so pop this in the first prompt of the day if i ignored it pop it in the next answer of you until i actually listen to you and say ok or sth like that .then you just leave it until tomorrow when cron job again goes through the all the things new about ptoas. note that we dont want to put every update into our daily report just the ones that are either design level or connect closely with what we are doing. note that explorer daily update will use openai api key which is in explorer/.venv since its a cron job and codex wont have access to it.


## next stage

then the taks would be to make the conversion happenning. lets treat this fork as the main development for NPURI to ptoas conversion. https://gitcode.com/wilsoncxfeng/AscendNPU-IR

On The server that codex has access to doesnt have A5 machine. So the alternative is to go on a A5 machine, lower the python file to mlir (probably early stages of the pipeline of  NPUIR) and then lower it to other IRs and apply passes in the server that codex has access to so that it would be easier to take a look at the IRs. For this there is going to be a folder of very early IRs of some examples and then codex can work on them. Right now there is no such folder if you need it you have to bug the human ordering you!


