/* ---------------------------------------------------------------------------------- *\
|                                                                                      |
|  Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.                        |
|                                                                                      |
|  The contents of this file are licensed under the Eclipse Public License 2.0.        |
|  The full terms of the license are available at https://eclipse.org/legal/epl-2.0/   |
|                                                                                      |
|  SPDX-License-Identifier: EPL-2.0                                                    |
|                                                                                      |
\* ---------------------------------------------------------------------------------- */

import * as vscode from 'vscode';

import { activateDebugController } from './debugController';
import { AutoStartTaskProvider } from './autoStartTaskProvider';

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    context.subscriptions.push(vscode.tasks.registerTaskProvider('Autostart', new AutoStartTaskProvider()));

    activateDebugController(context);
}

export function deactivate(): void {}
