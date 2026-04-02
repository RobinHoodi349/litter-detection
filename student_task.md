# Your task

> Your mission, should you choose to accept it...

## Aim

Build a litter detection system based on the proposed training history. This system should use images from the robodog and detect litter in it.

- While the operator controls the robot the dog should make some noise, if it detects litter.
- The system should be better than the proposed baseline and operate in realtime on the robodog hardware.
- The system should offer a possibility to identify and investigate possible wrong litter detections.

Reminder:

- Document the process and usage of AI during the lab task

Assumptions:

- litter can only be on the ground
- litter has a sufficient size (amount of pixel)

## Sensor

- Use either the image from the robodog camera or the image from the realsense

## Guardrails

- track the experiments with mlflow

## Zenoh Kickstart

We use zenoh as router. To start it as container use:

```bash
docker run --init -p 7447:7447/tcp -p 8000:8000/tcp eclipse/zenoh
```

Basic Tutorial for zenoh:

- Getting started with zenoh: https://zenoh.io/docs/getting-started/first-app/

## Idea

1. Understand the provided repository and steps taken by the system
   1. > What is missing? Where are 
2. Understand the approach of automated research and try to adapt it (everything allowed)
3. Prepare the model for inference on the Jetson hardware using tensorRT
4. Tune the system by adding additional perception approaches like an open word object detector or a different fine tuning approach.
