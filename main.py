import translator as tr
# coding: utf8


def main():


    test = tr.translator('ru')
    x = test.csv_reading('emails.csv')
    k = 1
    for i in x:
        print(k)
        k += 1
        text = test.text_parsing(i)
        # text = "New sign-in from Safari on Mac\r\n\r\n\r\n\r\nHi Support,\r\nYour Google account email@server.com was just used to sign in from\r\nSafari on Mac.\r\n\r\nSupport Contact\r\nemail@server.com\r\n\r\nMac\r\nMonday, 12 December 2016 11:41 (Eastern European Standard Time)\r\nKiev, Ukraine*\r\nSafari*Don't recognise this activity?*\r\nReview your recently used devices\r\n\u003chttps://accounts.google.com/AccountChooser?Email=email@server.com\u0026continue=https://security.google.com/settings/security/activity/nt/1481535716000?rfn%3D31%26rfnc%3D1%26et%3D0%26asae%3D2\u003e\r\nnow.\r\n\r\nWhy are we sending this? We take security very seriously and we want to\r\nkeep you in the loop on important actions in your account.\r\nWe were unable to determine whether you have used this browser or device\r\nwith your account before. This can happen when you sign in for the first\r\ntime on a new computer, phone or browser, when you use your browserвЂ™s\r\nincognito or private browsing mode or clear your cookies or when somebody\r\nelse is accessing your account.\r\n\r\nBest,\r\nThe Google Accounts team\r\n\r\n\r\n\r\n*The location is approximate and determined by the IP address it was coming\r\nfrom.\r\n\r\nThis email can't receive replies. To give us feedback on this alert, click\r\nhere\r\n\u003chttps://support.google.com/accounts/contact/device_alert_feedback?hl=en-GB\u003e\r\n.\r\nFor more information, visit the Google Accounts Help Centre\r\n\u003chttps://support.google.com/accounts/answer/2733203\u003e.\r\n\r\n\r\n\r\nYou have received this mandatory email service announcement to update you\r\nabout important changes to your Google product or account.\r\nВ© 2016 Google Inc., 1600 Amphitheatre Parkway, Mountain View, CA 94043, USA\r\n"
        tr_text = test.translating(text)
        test.writing('translation.docx', text, tr_text)
        print("wr")
        if k == 10:
            return


if __name__ == '__main__':
    main()
