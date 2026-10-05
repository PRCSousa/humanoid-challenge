# Training a VLA model to move a robotic arm with my own hands (literally)

In this project, I fine-tuned a VLA model using my own collection of videos and instructions (like "move left" or "rotate clockwise") with the objective of manipulating a robotic arm on customs instructions that I created, with my hand as a the teacher.

## How it began

My background as a researcher is in reinforcement learning, and naturally, I've always kept up with the current advancements in the domain, being the most interesting one to me VLAs. I feel like this technology, besides the wonders of an LLM, is the first actual 'sci-fi'-esque technology that will allow a computer to come to life and produce general-purpose machines.

Thing is, I've never built one, between finishing my Master's and working as a researcher, [Humanoid's](https://thehumanoid.ai/) provided me with the perfect environment to experiment and learn a bit more about this. Their challenge was to drive a robotic arm in a simulation env, and he main constraint was for us to use data that was collected by ourselves to do so. Time to get to work.

### The Plan

The idea behind what I will be doing is simple:
1. Create a set of simple instructions of things I want to make the robotic arm do.
2. Record myself doing it with my hand.
3. Convert hand into robot actions.
4. Replay what my hand did, this time on the simulator.
5. Fine-tune a VLA model with this data.

## Data Collection

This was the most straight-forward part of the project (at least the collection itself, we'll get to that). After some iterations of what I wanted to train on, I collected approximatedly 30 videos of me doing 7 different actions, slightly changing initial conditions to add some variety, you can see the data in [HuggingFace](https://huggingface.co/datasets/ReAscalon/humanoid_move_thing).

Some instructions include:
- Clapping the gripper;
- Rotating clockwise and counterclockwise;
- Moving in all directions;

Observe some clapping:

<p align="center">
<img src="imgs/clapping_hands.gif" style='width: 70%'/>
</p>

## Hand Tracking

The example on the original post used AprilTags as a visual landmark to align the simulation and the video fields, allowing us to easily calculate the relative positions between the camera and the robotic arm.

As I though about the challenges behind embodied AI, I deliberatedly chose to not use any kind of field alignment technique, and instead derive everything I needed from the environment itself (specifically, my own hand as the reference). This, of course, comes with a huge challenge, which no questions asked, was the biggest hurdle that I faced when developing this idea.

### Converting video to numbers

With these videos in hand (ha!), now I'd have to convert whatever my hand was doing into some useful data. After some research, [MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker), a library made specifically for image recognition, allowed us to collect positional information on 21 landmarks throughout our hand and wrist, as seen here:

<p align="center">
<img src="imgs/hand_connections.png" style='width: 70%'/>
</p>

And after a bit of trial and error, I got this going:

<p align="center">
<img src="imgs/tracked_clapping.gif"style='width: 70%'/>
</p>

Fantastic. Note the line connecting the thumb and index, this will be the main way of indicating if I am gripping something or not, and here is where the biggest challenge began.

#### Going forwards, and going upwards

Going from 2D video to 3D motion is quite hard, in the end we are operating with one dimension less than we need. Pixels move in two axes on a 2D video, and this does not provide us with any sense of depth, so moving my hand forwards produces the same change as rising it, a positive change in the Y axis, making 3D movement ambiguous in that scenario.

Going back to the original example, the use of AprilTags in the table allows us to calculate the geometry that locates the camera position relative to the table, and from here we can calculate coordinates and motion analytically through the change in video in relation to the camera position. And again, my objective was to do this without these markers, the challenge gave a lot of focus on data, so I wanted data to be my main concern.

My first instinct was to use MediaPipe's provided depth estimation, and after trying it for a while and seeing what kind of data it produced, I abandoned it, it was failing to distinguish between Y movement and Z movement. So, I tried to use my hand as a scale.

By making some assumptions in my hand size, I could use the distance between landmarks as a method to estimate depth, if my hand gets smaller, the landmarks get closer, which means the hand is further away. It did indeed work, now movement was being considerably better in terms of distinguishing both axes, but still nowhere near optimal. The problem in this approach wasn't all that opaque either, as hand position and perspective plays a huge role in this kind of calculations:

<p align="center">
<img src="imgs/perspective.png" style='width: 70%'/>
</p>

So I had to search for something a bit more intricate, and that's when I read about Perspective-n-Point (PnP). Normally, PnP is used to estimate a camera's position using a set of known 3D points, and their corresponding 2D projections, and this is really how AprilTags work in the end, we know the physical size of the tag and where their corners sit, and we have their 2D video projections, so we can estimate the camera position.

<p align="center">
<img src="imgs/PnP.png"style='width: 70%'/>
</p>

But, if we look at what we need to solve this problem, we actually (kinda) have everything we need to solve for it. Our hands, in terms of scale, do not really change all that much, and the palm specifically can't bend like our fingers, the palm is a rigid body, and we can estimate roughly their 3D positions (if we assume the palm landmarks are all coplanar to each other).

So, instead of calculating the relative position of the camera to our palm, we instead calculate the relative position of the palm to our camera! 

This did work, not perfectly, but planar Y Z movement became much less ambiguous when compared to the previous iterations of depth estimation.


### Preparing our data

Now that we can convert 2D video to hand positions, we will make use of the change in position of my hand to produce data, so my hand movement will be expressed as [dx, dy, dz, drx, dry, drz, gripper].

As I am human and as calculations go, the data had a lot of jitter, and when I looked at the initial retargetings inside the simulator, the robotic arm shaked a lot, and this would introduce a lot of problems in terms of precision if I were to train the VLA on larger tasks where precision was key. So, across all the data, I computed an EMA to smooth everything out.

Another important change was calculating the hand position in relation to the camera angle, so by using an approximation of how tilted the camera was, we could calculate the rotation matrix to align our data coordinates to the coordinates of the simulator (aligned in relation to the table).

In terms of the gripper, initially I looked rougly at what would be decent values to delineate if my thumb-index distance meant opened or closed, but this was very unstable, so I opted to calculate a kmeans per video to find a decent threshold between setting the gripper as opened or closed. In hindsight, I assumed the gripper to be binary, which led me to try and find the threshold in the continuous data, when I could've just map the max and min of each clip to [-1, 1] respectively.

Finally, now we just had to convert this data into LIBERO's convention, which was very straightforward, as it uses the same convention, and just required some scaling.

<p align="center">
<img src="imgs/sim2real.gif"style='width: 100%'/>
</p>

## Results

## Reproducibility

## References
 - [Solving egl-probe](https://github.com/huggingface/lerobot/issues/105)
 - [And hf-egl-probe](https://github.com/huggingface/lerobot/issues/3397)
 - [MediaPipe Documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)

---

This project was develped towards the Robot Learning Research Intership @ [Humanoid](https://thehumanoid.ai/).
