import yaml

from titans.email import SendEmails


def subscribe_test():
    SendEmails(**yaml.safe_load(open('email/test/config.yaml', 'r')))


if __name__ == '__main__':
    subscribe_test()
