from metaworld.policies import (
    SawyerAssemblyV2Policy,
    SawyerBasketballV2Policy,
    SawyerBinPickingV2Policy,
    SawyerBoxCloseV2Policy,
    SawyerButtonPressTopdownV2Policy,
    SawyerButtonPressTopdownWallV2Policy,
    SawyerButtonPressV2Policy,
    SawyerButtonPressWallV2Policy,
    SawyerCoffeeButtonV2Policy,
    SawyerCoffeePullV2Policy,
    SawyerCoffeePushV2Policy,
    SawyerDialTurnV2Policy,
    SawyerDisassembleV2Policy,
    SawyerDoorCloseV2Policy,
    SawyerDoorLockV2Policy,
    SawyerDoorOpenV2Policy,
    SawyerDoorUnlockV2Policy,
    SawyerDrawerCloseV2Policy,
    SawyerDrawerOpenV2Policy,
    SawyerFaucetCloseV2Policy,
    SawyerFaucetOpenV2Policy,
    SawyerHammerV2Policy,
    SawyerHandInsertV2Policy,
    SawyerHandlePressSideV2Policy,
    SawyerHandlePressV2Policy,
    SawyerHandlePullSideV2Policy,
    SawyerHandlePullV2Policy,
    SawyerLeverPullV2Policy,
    SawyerPegInsertionSideV2Policy,
    SawyerPegUnplugSideV2Policy,
    SawyerPickOutOfHoleV2Policy,
    SawyerPickPlaceV2Policy,
    SawyerPickPlaceWallV2Policy,
    SawyerPlateSlideBackSideV2Policy,
    SawyerPlateSlideBackV2Policy,
    SawyerPlateSlideSideV2Policy,
    SawyerPlateSlideV2Policy,
    SawyerPushBackV2Policy,
    SawyerPushV2Policy,
    SawyerPushWallV2Policy,
    SawyerReachV2Policy,
    SawyerReachWallV2Policy,
    SawyerShelfPlaceV2Policy,
    SawyerSoccerV2Policy,
    SawyerStickPullV2Policy,
    SawyerStickPushV2Policy,
    SawyerSweepIntoV2Policy,
    SawyerSweepV2Policy,
    SawyerWindowCloseV2Policy,
    SawyerWindowOpenV2Policy,
)

policies = {
    "assembly-v2": (SawyerAssemblyV2Policy, 'pick up a nut and place it onto a peg'),
    "basketball-v2": (SawyerBasketballV2Policy, 'dunk the basketball into the basket'),
    "bin-picking-v2": (SawyerBinPickingV2Policy, 'grasp the puck from one bin and place it into another bin'),
    "box-close-v2": (SawyerBoxCloseV2Policy, 'grasp the cover and close the box with it'),
    "button-press-topdown-v2": (SawyerButtonPressTopdownV2Policy, 'press a button from the top'),
    "button-press-topdown-wall-v2": (SawyerButtonPressTopdownWallV2Policy, 'bypass a wall and press a button from the top'),
    "button-press-v2": (SawyerButtonPressV2Policy, 'press a button'),
    "button-press-wall-v2": (SawyerButtonPressWallV2Policy, 'bypass a wall and press a button'),
    "coffee-button-v2": (SawyerCoffeeButtonV2Policy, 'push a button on the coffee machine'),
    "coffee-pull-v2": (SawyerCoffeePullV2Policy, 'pull a mug from a coffee machine to the green goal'),
    "coffee-push-v2": (SawyerCoffeePushV2Policy, 'push a mug under a coffee machine to the red goal'),
    "dial-turn-v2": (SawyerDialTurnV2Policy, 'rotate a dial counter clockwise'),
    "disassemble-v2": (SawyerDisassembleV2Policy, 'pick a nut out of the a peg'),
    "door-close-v2": (SawyerDoorCloseV2Policy, 'close a door with a revolving joint'),
    "door-lock-v2": (SawyerDoorLockV2Policy, 'lock the door by rotating the lock counter clockwise'),
    "door-open-v2": (SawyerDoorOpenV2Policy, 'open a door with a revolving joint'),
    "door-unlock-v2": (SawyerDoorUnlockV2Policy, 'unlock the door by rotating the lock clockwise'),
    "drawer-close-v2": (SawyerDrawerCloseV2Policy, 'push and close a drawer'),
    "drawer-open-v2": (SawyerDrawerOpenV2Policy, 'open a drawer'),
    "faucet-close-v2": (SawyerFaucetCloseV2Policy, 'rotate the faucet counter clockwise'),
    "faucet-open-v2": (SawyerFaucetOpenV2Policy, 'rotate the faucet clockwise'),
    "hammer-v2": (SawyerHammerV2Policy, 'hammer a screw on the wall'),
    "hand-insert-v2": (SawyerHandInsertV2Policy, 'insert the gripper into a hole'),
    "handle-press-side-v2": (SawyerHandlePressSideV2Policy, 'press a handle down sideways'),
    "handle-press-v2": (SawyerHandlePressV2Policy, 'press a handle down'),
    "handle-pull-v2": (SawyerHandlePullV2Policy, 'pull a handle up'),
    "handle-pull-side-v2": (SawyerHandlePullSideV2Policy, 'pull a handle up sideways'),
    "peg-insert-side-v2": (SawyerPegInsertionSideV2Policy, 'insert a peg sideways'),
    "lever-pull-v2": (SawyerLeverPullV2Policy, 'pull a lever up'),
    "peg-unplug-side-v2": (SawyerPegUnplugSideV2Policy, 'unplug a peg sideways'),
    "pick-out-of-hole-v2": (SawyerPickOutOfHoleV2Policy, 'pick up a puck from a hole'),
    "pick-place-v2": (SawyerPickPlaceV2Policy, 'pick and place a puck to the blue goal'),
    "pick-place-wall-v2": (SawyerPickPlaceWallV2Policy, 'pick a puck, bypass a wall and place the puck to the blue goal'),
    "plate-slide-back-side-v2": (SawyerPlateSlideBackSideV2Policy, 'get a plate from the cabinet sideways to the red goal'),
    "plate-slide-back-v2": (SawyerPlateSlideBackV2Policy, 'get a plate from the cabinet to the red goal'),
    "plate-slide-side-v2": (SawyerPlateSlideSideV2Policy, 'slide a plate into a cabinet sideways'),
    "plate-slide-v2": (SawyerPlateSlideV2Policy, 'slide a plate into a cabinet'),
    "reach-v2": (SawyerReachV2Policy, 'reach the red sphere goal position'),
    "reach-wall-v2": (SawyerReachWallV2Policy, 'bypass a wall and reach the red sphere goal position'),
    "push-back-v2": (SawyerPushBackV2Policy, 'pull a puck to a goal'),
    "push-v2": (SawyerPushV2Policy, 'push the puck to a goal'),
    "push-wall-v2": (SawyerPushWallV2Policy, 'bypass a wall and push a puck to a goal'),
    "shelf-place-v2": (SawyerShelfPlaceV2Policy, 'pick and place a puck onto a shelf'),
    "soccer-v2": (SawyerSoccerV2Policy, 'kick a soccer into the goal'),
    "stick-pull-v2": (SawyerStickPullV2Policy, 'grasp a stick and pull a box with the stick'),
    "stick-push-v2": (SawyerStickPushV2Policy, 'grasp a stick and push a box using the stick'),
    "sweep-into-v2": (SawyerSweepIntoV2Policy, 'sweep a puck into a hole'),
    "sweep-v2": (SawyerSweepV2Policy, 'sweep a puck off the table'),
    "window-close-v2": (SawyerWindowCloseV2Policy, 'push and close a window'),
    "window-open-v2": (SawyerWindowOpenV2Policy, 'push and open a window'),
}

ML45_test_tasks = ['bin-picking-v2', 'box-close-v2', 'hand-insert-v2', 'door-lock-v2', 'door-unlock-v2']