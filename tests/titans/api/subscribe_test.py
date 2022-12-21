from titansemail import SendEmails
import yaml


def subscribe_test():
    SendEmails(**yaml.safe_load(open('email/config.yaml', 'r')))


if __name__ == '__main__':
    subscribe_test()
