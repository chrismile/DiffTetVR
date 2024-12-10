# BSD 2-Clause License
#
# Copyright (c) 2024, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import getpass
import itertools
import pathlib
import subprocess
import html
import smtplib
import ssl
from email.message import EmailMessage
from email.headerregistry import Address
from email.utils import formatdate


def send_mail(
        sender_name, sender_email_address, user_name, password,
        recipient_name, recipient_email_address,
        subject, message_text_raw, message_text_html):
    if sender_email_address.endswith('@in.tum.de'):
        smtp_server = 'mail.in.tum.de'
        port = 587  # STARTTLS
    elif sender_email_address.endswith('@tum.de'):
        smtp_server = 'postout.lrz.de'
        port = 587  # STARTTLS
    else:
        raise Exception(f'Error: Unexpected provider in e-mail address {sender_email_address}!')

    context = ssl.create_default_context()
    server = smtplib.SMTP(smtp_server, port)
    server.ehlo()
    server.starttls(context=context)
    server.ehlo()
    server.login(user_name, password)

    message = EmailMessage()
    message['Subject'] = subject
    message['From'] = Address(display_name=sender_name, addr_spec=sender_email_address)
    message['To'] = Address(display_name=recipient_name, addr_spec=recipient_email_address)
    message['Date'] = formatdate(localtime=True)
    message.set_content(message_text_raw)
    message.add_alternative(message_text_html, subtype='html')

    server.sendmail(sender_email_address, recipient_email_address, message.as_string())

    server.quit()


def escape_html(s):
    s_list = html.escape(s, quote=False).splitlines(True)
    s_list_edit = []
    for se in s_list:
        se_notrail = se.lstrip()
        new_se = se_notrail
        for i in range(len(se) - len(se_notrail)):
            new_se = '&nbsp;' + new_se
        s_list_edit.append(new_se)
    s = ''.join(s_list_edit)
    return s.replace('\n', '<br/>\n')


preshaded_path = os.path.join(pathlib.Path.home(), 'Programming/C++/Correrender/Data/VolumeDataSets/preshaded')
regular_grids_path = '/mnt/data/Flow/Scalar'
if not os.path.isdir(regular_grids_path):
    regular_grids_path = os.path.join(pathlib.Path.home(), 'datasets/Scalar')
if not os.path.isdir(regular_grids_path):
    regular_grids_path = os.path.join(pathlib.Path.home(), 'datasets/Flow/Scalar')
if not os.path.isdir(regular_grids_path):
    regular_grids_path = f'/media/{getpass.getuser()}/Elements/Datasets/Scalar'

if os.name == 'nt':
    python_cmd = 'python'
else:
    python_cmd = 'python3'
commands = []

# (1) Test case for different color LR.
for lr_col in [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14]:
    commands.append([
        python_cmd, 'train.py',
        '--name', f'tooth_color_{lr_col}',
        '--out_dir', os.path.join(pathlib.Path.home(), 'datasets/Tet/Test'),
        '--attenuation', '100.0',
        '--lr_col', str(lr_col),
        '--lr_pos', '0.0',
        '--init_grid_path', os.path.join(preshaded_path, 'tooth_uniform.bintet'),
        '--gt_grid_path', os.path.join(regular_grids_path, 'Tooth [256 256 161](CT)', 'tooth_cropped.dat'),
        '--gt_tf', 'Tooth3Gauss.xml',
        '--record_video', '--save_statistics',
        '--cam_sample_method', 'replicate_cpp',
    ])

# (2) Test case for CTF with different regularization beta.
for tet_reg_beta in [1.0, 10.0, 100.0, 1000.0]:
    commands.append([
        python_cmd, 'train.py',
        '--name', f'tooth_ctf_reg_beta_{tet_reg_beta}',
        '--out_dir', os.path.join(pathlib.Path.home(), 'datasets/Tet/Test'),
        '--attenuation', '100.0',
        '--lr_col', '0.06',
        '--lr_pos', '0.00001',
        '--gt_grid_path', os.path.join(regular_grids_path, 'Tooth [256 256 161](CT)', 'tooth_cropped.dat'),
        '--gt_tf', 'Tooth3Gauss.xml',
        '--record_video', '--save_statistics',
        '--coarse_to_fine', '--max_num_tets', '100000', '--fix_boundary', '--splits_ratio', '0.05',
        '--tet_regularizer', '--tet_reg_lambda', '1000000.0', '--tet_reg_softplus_beta', str(tet_reg_beta),
        '--cam_sample_method', 'replicate_cpp',
    ])

# (3) Test case for CTF with different regularization lambdas.
for tet_reg_lambda in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
    commands.append([
        python_cmd, 'train.py',
        '--name', f'tooth_ctf_reg_lambda_{tet_reg_lambda}',
        '--out_dir', os.path.join(pathlib.Path.home(), 'datasets/Tet/Test'),
        '--attenuation', '100.0',
        '--lr_col', '0.06',
        '--lr_pos', '0.00001',
        '--gt_grid_path', os.path.join(regular_grids_path, 'Tooth [256 256 161](CT)', 'tooth_cropped.dat'),
        '--gt_tf', 'Tooth3Gauss.xml',
        '--record_video', '--save_statistics',
        '--coarse_to_fine', '--max_num_tets', '100000', '--fix_boundary', '--splits_ratio', '0.05',
        '--tet_regularizer', '--tet_reg_lambda', str(tet_reg_lambda), '--tet_reg_softplus_beta', '100.0',
        '--cam_sample_method', 'replicate_cpp',
    ])

# (4) Test case for CTF with regularizer and position gradients with different learning rates.
for lr_pos in [0.000001, 0.000005, 0.00001, 0.00005, 0.0001]:
    commands.append([
        python_cmd, 'train.py',
        '--name', f'tooth_ctf_pos_{lr_pos}',
        '--out_dir', os.path.join(pathlib.Path.home(), 'datasets/Tet/Test'),
        '--attenuation', '100.0',
        '--lr_col', '0.06',
        '--lr_pos', str(lr_pos),
        '--gt_grid_path', os.path.join(regular_grids_path, 'Tooth [256 256 161](CT)', 'tooth_cropped.dat'),
        '--gt_tf', 'Tooth3Gauss.xml',
        '--record_video', '--save_statistics',
        '--coarse_to_fine', '--max_num_tets', '100000', '--fix_boundary', '--splits_ratio', '0.05',
        '--tet_regularizer', '--tet_reg_lambda', '10.0', '--tet_reg_softplus_beta', '100.0',
        '--cam_sample_method', 'replicate_cpp',
    ])

# (5) Test case for CTF with regularizer and position gradients with different learning rates.
for num_tets in [10000, 30000, 100000, 500000, 1000000]:
    commands.append([
        python_cmd, 'train.py',
        '--name', f'tooth_ctf_num_tets_{num_tets}',
        '--out_dir', os.path.join(pathlib.Path.home(), 'datasets/Tet/Test'),
        '--attenuation', '100.0',
        '--lr_col', '0.06',
        '--lr_pos', '0.00001',
        '--gt_grid_path', os.path.join(regular_grids_path, 'Tooth [256 256 161](CT)', 'tooth_cropped.dat'),
        '--gt_tf', 'Tooth3Gauss.xml',
        '--record_video', '--save_statistics',
        '--coarse_to_fine', '--max_num_tets', str(num_tets), '--fix_boundary', '--splits_ratio', '0.05',
        '--tet_regularizer', '--tet_reg_lambda', '10.0', '--tet_reg_softplus_beta', '100.0',
        '--cam_sample_method', 'replicate_cpp',
    ])

commands.append([python_cmd, 'eval.py'])


if __name__ == '__main__':
    shall_send_email = True
    pwd_path = os.path.join(pathlib.Path.home(), 'Documents', 'mailpwd.txt')
    use_email = pathlib.Path(pwd_path).is_file()
    if use_email:
        with open(pwd_path, 'r') as file:
            lines = [line.rstrip() for line in file]
            sender_name = lines[0]
            sender_email_address = lines[1]
            user_name = lines[2]
            password = lines[3]
            recipient_name = lines[4]
            recipient_email_address = lines[5]

    for command in commands:
        print(f"Running '{' '.join(command)}'...")
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = proc.communicate()
        proc_status = proc.wait()
        if proc_status != 0:
            if os.name == 'nt':
                stderr_string = err.decode('latin-1')
                stdout_string = output.decode('latin-1')
            else:
                stderr_string = err.decode('utf-8')
                stdout_string = output.decode('utf-8')

            if use_email:
                message_text_raw = f'The following command failed with code {proc_status}:\n'
                message_text_raw += ' '.join(command) + '\n\n'
                message_text_raw += '--- Output from stderr ---\n'
                message_text_raw += stderr_string
                message_text_raw += '---\n\n'
                message_text_raw += '--- Output from stdout ---\n'
                message_text_raw += stdout_string
                message_text_raw += '---'

                message_text_html = \
                    '<html>\n<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"></head>\n<body>\n'
                message_text_html += f'The following command failed with code {proc_status}:<br/>\n'
                message_text_html += ' '.join(command) + '<br/><br/>\n\n'
                message_text_html += '<font color="red" style="font-family: \'Courier New\', monospace;">\n'
                message_text_html += '--- Output from stderr ---<br/>\n'
                message_text_html += escape_html(stderr_string)
                message_text_html += '---</font>\n<br/><br/>\n\n'
                message_text_html += '<font style="font-family: \'Courier New\', monospace;">\n'
                message_text_html += '--- Output from stdout ---<br/>\n'
                message_text_html += escape_html(stdout_string)
                message_text_html += '---</font>\n'
                message_text_html += '</body>\n</html>'

                if shall_send_email:
                    send_mail(
                        sender_name, sender_email_address, user_name, password,
                        recipient_name, recipient_email_address,
                        'Error while generating images', message_text_raw, message_text_html)

            print('--- Output from stdout ---')
            print(stdout_string.rstrip('\n'))
            print('---\n')
            print('--- Output from stderr ---', file=sys.stderr)
            print(stderr_string.rstrip('\n'), file=sys.stderr)
            print('---', file=sys.stderr)
            sys.exit(1)
            #raise Exception(f'Process returned error code {proc_status}.')
        elif not shall_send_email:
            stderr_string = err.decode('utf-8')
            stdout_string = output.decode('utf-8')
            print('--- Output from stdout ---')
            print(stdout_string.rstrip('\n'))
            print('---\n')
            print('--- Output from stderr ---', file=sys.stderr)
            print(stderr_string.rstrip('\n'), file=sys.stderr)
            print('---', file=sys.stderr)

    message_text_raw = 'run.py finished successfully'
    message_text_html = \
        '<html>\n<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"></head>\n<body>\n'
    message_text_html += 'run.py finished successfully'
    message_text_html += '</body>\n</html>'
    if shall_send_email:
        send_mail(
            sender_name, sender_email_address, user_name, password,
            recipient_name, recipient_email_address,
            'run.py finished successfully', message_text_raw, message_text_html)
    print('Finished.')
